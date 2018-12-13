import os
import signal
import sys
from gurobipy import *
import multiprocessing
import time

global mod

class TimeOut(Exception):
    pass


class net_in_LP:
    def __init__(self, LB, UB, layer_num, label, start):
        self.init_layer_num = layer_num
        self.last_layer_num = layer_num
        self.label = label
        self.start = start

        #bounds to model
        self.model = Model("myLP")
        self.model.setParam('OutputFlag', False)
        #self.model.setParam('Method', 1)
        #self.model.setParam('OptimalityTol',1e-2)
        #self.model.setParam('FeasibilityTol', 1e-2)
        self.last_layer = self.model.addVars(len(LB), name=str(layer_num) + "v", lb=LB, ub=UB)

        # the bounds before the last ReLu Layer
        self.last_bounds_LB = LB
        self.last_bounds_UB = UB
        self.T_limit = 1e10
        self.processes_used = max(multiprocessing.cpu_count() - 1, 2)
        print("the number of processes used is", self.processes_used)

    def add_ReLu(self):
        self.last_layer_num += 1
        self.last_bounds_LB, self.last_bounds_UB = self.go_to_box(approximative=True)
        n = len(self.last_layer)

        for k in range(n):
            if self.last_bounds_LB[k] < 0:
                if self.last_bounds_UB[k] <= 0:
                    self.last_layer[k] = self.model.addVar(lb=0, ub=0)
                else:
                    temp = self.model.addVar(lb=0, ub=GRB.INFINITY)
                    self.model.addLConstr(temp >= self.last_layer[k], str(self.last_layer_num) + "cc" + str(k))
                    lam = self.last_bounds_UB[k] / (self.last_bounds_UB[k] - self.last_bounds_LB[k])
                    d = -self.last_bounds_LB[k] * lam
                    self.model.addLConstr(temp <= lam * self.last_layer[k] + d, str(self.last_layer_num) + "ccc" + str(k))

                    self.last_layer[k] = temp

    def add_affine(self, weights, biases):
        self.last_layer_num += 1
        size = weights.shape
        n = size[1]
        m = size[0]
        assert(n == len(self.last_layer))
        assert(m == len(biases))

        h = self.model.addVars(m, name=str(self.last_layer_num) + "v", lb=-GRB.INFINITY, ub=GRB.INFINITY)

        # here we create the linear expression in a very efficient manner. but a ordered list of gurobi variables is needed
        vars = [var for (key, var) in sorted(self.last_layer.iteritems())]
        for j in range(m):
            fun = LinExpr(weights[j, :], vars)
            fun.addConstant(biases[j])
            self.model.addLConstr(h[j] == fun, name=str(self.last_layer_num) + "c")

        self.last_layer = h

    def start_job(self, i, approximative, MAXIMIZE):
        rend, wend = os.pipe()
        if MAXIMIZE:
            self.model.setObjective(self.last_layer[i], GRB.MAXIMIZE)
            if approximative:
                self.model.setParam('Cutoff', 0.0)
            else:
                self.model.setParam('Cutoff', -GRB.INFINITY)
        else:
            self.model.setObjective(self.last_layer[i], GRB.MINIMIZE)
            if approximative:
                self.model.setParam('Cutoff', 0.0)
            else:
                self.model.setParam('Cutoff', GRB.INFINITY)

        pid = os.fork()
        if pid == 0:
            # print("I am from the child", os.getpid())
            whand = os.fdopen(wend, 'w', 1)
            self.model.optimize()
            objective = self.model.getObjective()

            if self.model.Status == GRB.OPTIMAL:
                objective = objective.getValue()
            elif self.model.Status == GRB.CUTOFF:
                objective = 0.0
            elif self.model.Status == GRB.TIME_LIMIT:
                whand.write("TimeOut")
                raise TimeOut
            else:
                assert False

            whand.write(str(objective) + "\n")
            #sys.exit()
            os._exit(0)

        return pid, rend

    def go_to_box(self, approximative):
        jobs_left = self.processes_used
        progressbar = 0
        print("¦==============================¦")
        print("¦",end='', flush=True)
        n = len(self.last_layer)
        UB = []
        LB = []
        fd_ub = {}
        fd_lb = {}
        proc_ub = {}
        proc_lb = {}

        i = 0
        end = 0
        while True:
            if i < n and jobs_left > 0:
                proc_ub[i], fd_ub[i] = self.start_job(i, approximative, True)
                proc_lb[i], fd_lb[i] = self.start_job(i, approximative, False)

                jobs_left += -2
                #print("I am from the mother", os.getpid())
                i += 1

            if i == n or jobs_left <= 0:

                os.waitpid(proc_ub[end], 0)
                rhand = os.fdopen(fd_ub[end],'r',1)
                UB.append(float(rhand.readline()))
                rhand.close()

                os.waitpid(proc_lb[end], 0)
                rhand = os.fdopen(fd_lb[end],'r',1)
                LB.append(float(rhand.readline()))
                rhand.close()
                #os.kill(proc_lb[end], signal.SIGKILL)

                jobs_left += 2
                end += 1

                while progressbar < 30*end/n:
                    print("=", end='',flush=True)
                    progressbar += 1
            if end == n:
                del fd_ub
                del fd_lb
                break

        print('¦')
        return LB, UB

    def verify_label(self):
        #  this function is a performance improvement for the last layer bounds, it can prove that
        #  all other labels than label have a strictly smaller logit value than the label.
        #  can only be called for the last layer of the network
        #  it gives tighter bounds than the bounds comparison.

        # prepare the indices which are not the label
        r = list(range(len(self.last_layer)))
        del r[self.label]
        for i in r:
            self.model.setObjective(self.last_layer[i] - self.last_layer[self.label], GRB.MAXIMIZE)
            self.model.setParam('BestObjStop', 0.0)
            self.model.setParam('Cutoff', 0.0)
            # model.setParam('TimeLimit', 0.01)
            self.model.optimize()
            if self.model.Status == GRB.USER_OBJ_LIMIT:
                print("label logit (" + str(self.label) + ") can be larger than logit " + str(i))
                return False
            elif self.model.Status == GRB.OPTIMAL:
                obj = self.model.getObjective()
                obv = obj.getValue()
                print("logit " + str(i) + " - label logit (" + str(self.label) + ") <= " + str(obv))
                if obv >= 0:
                    return False
            elif self.model.Status == GRB.CUTOFF:
                print("logit " + str(i) + " is strictly smaller than the label logit (" + str(self.label) + ")")
            else:
                assert False

        return True
