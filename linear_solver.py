from gurobipy import *
import numpy as np
import time

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
        self.last_layer = self.model.addVars(len(LB), name=str(layer_num) + "v", lb=LB, ub=UB)

        # the bounds before the last ReLu Layer
        self.last_bounds_LB = LB
        self.last_bounds_UB = UB
        self.T_limit = 1e10

    def add_ReLu(self, LB=None, UB=None, fast=False, stop_t=None):
        # some bounds may be knownn from the elina solver
        if (UB is not None) and (LB is not None):
            assert(len(LB) == len(self.last_layer) and len(UB) == len(self.last_layer))
            self.model.setAttr("LB", self.last_layer, LB)
            self.model.setAttr("UB", self.last_layer, UB)

        if fast and (UB is not None) and (LB is not None):
            self.last_bounds_LB = LB
            self.last_bounds_UB = UB
        else:
            self.last_bounds_LB, self.last_bounds_UB = self.go_to_box(LB, UB, approximative=True, stop_t=stop_t)

        self.last_layer_num += 1
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
        #  print({i: e for i, e in enumerate(weights[1, :])})
        #  print(last_layer)
            # test = sorted(last_layer.iterkeys())
            # assert(test == list(range(n)))

        # here we create the linear expression in a very efficient manner. but a ordered list of gurobi variables is needed
        vars = [var for (key, var) in sorted(self.last_layer.iteritems())]
        for j in range(m):
            fun = LinExpr(weights[j, :], vars)
            fun.addConstant(biases[j])
            self.model.addLConstr(h[j] == fun, name=str(self.last_layer_num) + "c")

        self.last_layer = h

    def go_to_box(self, LB_in, UB_in, approximative, stop_t=None):
        t_start = time.time()
        n = len(self.last_layer)
        UB = []
        LB = []
        for i in range(n):
            if stop_t is None or stop_t > time.time():
                UB.append(self.find_one_bound(self.last_layer[i], True, approximative))
                LB.append(self.find_one_bound(self.last_layer[i], False, approximative))
            else:
                UB.append(UB_in[i])
                LB.append(LB_in[i])

        return LB, UB

    def find_one_bound(self, objective, MAXIMIZE, approximative):
        if MAXIMIZE:
            self.model.setObjective(objective, GRB.MAXIMIZE)
            if approximative:
                self.model.setParam('Cutoff', 0.0)
            else:
                self.model.setParam('Cutoff', -GRB.INFINITY)
        else:
            self.model.setObjective(objective, GRB.MINIMIZE)
            if approximative:
                self.model.setParam('Cutoff', 0.0)
            else:
                self.model.setParam('Cutoff', GRB.INFINITY)
        #  rest_time = self.T_limit - (time.time() - self.start)
        #  if rest_time < 0:
        #    raise TimeOut
        #  self.model.setParam('TimeLimit', rest_time)
        self.model.optimize()
        if self.model.Status == GRB.OPTIMAL:
            return objective.x
        elif self.model.Status == GRB.CUTOFF:
            return 0.0
        elif self.model.Status == GRB.TIME_LIMIT:
            raise TimeOut
        else:
            assert False

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



# def add_ReLu_slow(model, last_layer, layer_num):
#     LB, UB = go_to_box(model, last_layer)
#
#     n = len(last_layer)
#     r = model.addVars(n, name=str(layer_num) + "v", lb=-GRB.INFINITY, ub=GRB.INFINITY)
#     for k in range(n):
#         #  ik = model.getVarByName(str(layer_num-1) + "v[" + str(k) + "]")
#         if LB[k] >= 0:
#             model.addLConstr(r[k] == last_layer[k], str(layer_num) + "c" + str(k))
#         else:
#             if UB[k] <= 0:
#                 model.addLConstr(r[k] == 0, str(layer_num) + "c" + str(k))
#             else:
#                 model.addLConstr(r[k] >= 0, str(layer_num) + "c" + str(k))
#                 model.addLConstr(r[k] >= last_layer[k], str(layer_num) + "cc" + str(k))
#                 lam = UB[k] / (UB[k] - LB[k])
#                 d = -LB[k] * lam
#                 model.addLConstr(r[k] <= lam * last_layer[k] + d, str(layer_num) + "ccc" + str(k))
#     return model, r


#  This function was the attempt to model more tight constraints by adding a quadratic constraint
#  Unfortunately it is not solvable because the set is not convex anymore and the Q matrix not positive semidefinite
# def add_ReLu_precise(model, last_layer, layer_num):
#     LB, UB = go_to_box(model, last_layer)
#     n = len(last_layer)
#     r = model.addVars(n, name=str(layer_num) + "v", lb=-GRB.INFINITY, ub=GRB.INFINITY)
#     for k in range(n):
#         if LB[k] <= 0:
#             if UB[k] <= 0:
#                 last_layer[k] = 0
#             else:
#                 temp = model.addVar(lb=0, ub=GRB.INFINITY)
#                 #model.addLConstr(temp >= 0, str(layer_num) + "c" + str(k))
#                 model.addLConstr(temp >= last_layer[k], str(layer_num) + "cc" + str(k))
#                 #lam = UB[k] / (UB[k] - LB[k])
#                 #d = -LB[k] * lam
#                 #model.addLConstr(temp <= lam * last_layer[k] + d, str(layer_num) + "ccc" + str(k))
#                 zba =2*UB+LB
#                 #model.addConstr((last_layer[k] - LB[k])*(last_layer[k] - LB[k]) + (temp-ba[k])*(temp-ba[k]) >= ba[k]*ba[k])
#                 #model.addConstr(last_layer[k]*last_layer[k] - 2*LB[k]*last_layer[k] + LB[k]*LB[k] + temp*temp - 2*temp*zba + zba*zba >= zba)
#                 model.addConstr(last_layer[k] * last_layer[k] - 2*LB[k]*last_layer[k] + LB[k]*LB[k] + temp*temp - 2*temp*zba[k] + zba[k] * zba[k] >= zba[k])
#
#                 last_layer[k] = temp
#
#     return model, last_layer

