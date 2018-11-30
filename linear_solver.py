from gurobipy import *
import numpy as np


class net_in_LP:
    def __init__(self, LB, UB, layer_num, label):
        self.model, self.last_layer = bounds_to_model(LB, UB, layer_num)
        self.init_layer_num = layer_num
        self.last_layer_num = layer_num
        self.label = label

    def add_ReLu(self):
        self.last_layer_num += 1
        self.model, self.last_layer = add_ReLu(self.model, self.last_layer, self.last_layer_num)

    def add_affine(self, weights, biases):
        self.last_layer_num += 1
        self.model, self.last_layer = add_affine(self.model, weights, biases, self.last_layer, self.last_layer_num)

    def go_to_box(self):
        return go_to_box(self.model, self.last_layer)

    def verify_label(self):
        return verify_label(self.model, self.last_layer, self.label)


def verify_label(model, last_layer, label):
    #  this function is a performance improvement for the last layer bounds, it can prove that
    #  all other labels than label have a strictly smaller logit value than the label.
    #  can only be called for the last layer of the network
    #  it gives tighter bounds than the bounds comparison.

    # prepare the indices which are not the label
    r = list(range(len(last_layer)))
    del r[label]
    for i in r:
        model.setObjective(last_layer[i] - last_layer[label], GRB.MAXIMIZE)
        model.setParam('BestObjStop', 0.0)
        model.setParam('Cutoff', 0.0)
        #model.setParam('TimeLimit', 0.01)
        model.optimize()
        if model.Status == GRB.USER_OBJ_LIMIT:
            print("label logit (" + str(label) + ") can be larger than logit " + str(i))
            return False
        if model.Status == GRB.OPTIMAL:
            obj = model.getObjective()
            obv = obj.getValue()
            print("logit " + str(i) + " - label logit (" + str(label) + ") <= " + str(obv))
            if obv >= 0:
                return False
        if model.Status == GRB.CUTOFF:
            print("logit " + str(i) + " is strictly smaller than the label logit (" + str(label) + ")")

    return True


def go_to_box(model, last_layer):
    n = len(last_layer)
    UB = np.zeros(n)
    LB = np.zeros(n)
    for i in range(n):
        model.setObjective(last_layer[i], GRB.MAXIMIZE)
        # model.setParam('TimeLimit',.1)
        model.setParam('Cutoff',0.0)
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            UB[i] = last_layer[i].x
        elif model.Status == GRB.CUTOFF:
            UB[i] = 0.0
        else:
            assert(False)

        model.setObjective(last_layer[i], GRB.MINIMIZE)
        model.setParam('Cutoff', 0.0)
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            LB[i] = last_layer[i].x
        elif model.Status == GRB.CUTOFF:
            LB[i] = 0.0
        else:
            assert(False)

    return LB, UB


def bounds_to_model(LB, UB, layer_num):
    model = Model("myLP")
    model.setParam('OutputFlag', False)
    last_layer = model.addVars(len(LB), name=str(layer_num) + "v", lb=LB, ub=UB)
    return model, last_layer


def add_ReLu_slow(model, last_layer, layer_num):
    LB, UB = go_to_box(model, last_layer)

    n = len(last_layer)
    r = model.addVars(n, name=str(layer_num) + "v", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    for k in range(n):
        #  ik = model.getVarByName(str(layer_num-1) + "v[" + str(k) + "]")
        if LB[k] >= 0:
            model.addLConstr(r[k] == last_layer[k], str(layer_num) + "c" + str(k))
        else:
            if UB[k] <= 0:
                model.addLConstr(r[k] == 0, str(layer_num) + "c" + str(k))
            else:
                model.addLConstr(r[k] >= 0, str(layer_num) + "c" + str(k))
                model.addLConstr(r[k] >= last_layer[k], str(layer_num) + "cc" + str(k))
                lam = UB[k] / (UB[k] - LB[k])
                d = -LB[k] * lam
                model.addLConstr(r[k] <= lam * last_layer[k] + d, str(layer_num) + "ccc" + str(k))
    return model, r


def add_ReLu(model, last_layer, layer_num):
    LB, UB = go_to_box(model, last_layer)
    n = len(last_layer)

    for k in range(n):
        if LB[k] < 0:
            if UB[k] <= 0:
                last_layer[k] = model.addVar(lb=0, ub=0)
            else:
                temp = model.addVar(lb=0, ub=GRB.INFINITY)
                model.addLConstr(temp >= last_layer[k], str(layer_num) + "cc" + str(k))
                lam = UB[k] / (UB[k] - LB[k])
                d = -LB[k] * lam
                model.addLConstr(temp <= lam * last_layer[k] + d, str(layer_num) + "ccc" + str(k))

                last_layer[k] = temp

    return model, last_layer


#  This function was the attempt to model more tight constraints by adding a quadratic constraint
#  Unfortunately it is not solvable because the set is not convex anymore and the Q matrix not positive semidefinite
def add_ReLu_precise(model, last_layer, layer_num):
    LB, UB = go_to_box(model, last_layer)
    n = len(last_layer)
    r = model.addVars(n, name=str(layer_num) + "v", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    for k in range(n):
        if LB[k] <= 0:
            if UB[k] <= 0:
                last_layer[k] = 0
            else:
                temp = model.addVar(lb=0, ub=GRB.INFINITY)
                #model.addLConstr(temp >= 0, str(layer_num) + "c" + str(k))
                model.addLConstr(temp >= last_layer[k], str(layer_num) + "cc" + str(k))
                #lam = UB[k] / (UB[k] - LB[k])
                #d = -LB[k] * lam
                #model.addLConstr(temp <= lam * last_layer[k] + d, str(layer_num) + "ccc" + str(k))
                zba =2*UB+LB
                #model.addConstr((last_layer[k] - LB[k])*(last_layer[k] - LB[k]) + (temp-ba[k])*(temp-ba[k]) >= ba[k]*ba[k])
                #model.addConstr(last_layer[k]*last_layer[k] - 2*LB[k]*last_layer[k] + LB[k]*LB[k] + temp*temp - 2*temp*zba + zba*zba >= zba)
                model.addConstr(last_layer[k] * last_layer[k] - 2*LB[k]*last_layer[k] + LB[k]*LB[k] + temp*temp - 2*temp*zba[k] + zba[k] * zba[k] >= zba[k])

                last_layer[k] = temp

    return model, last_layer


def add_affine(model, weights, biases, last_layer, layer_num):
    size = weights.shape
    #n = size[1]
    m = size[0]
    #assert(n == len(last_layer))
    #assert(m == len(biases))

    h = model.addVars(m, name=str(layer_num) + "v", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    #  print({i: e for i, e in enumerate(weights[1, :])})
    #  print(last_layer)
    slowcalc = False
    if slowcalc:
        model.addConstrs(
         (h[j] == biases[j] + last_layer.prod({i: e for i, e in enumerate(weights[j, :])}) for j in range(m)),
         name=str(layer_num) + "c")
    else:
        #test = sorted(last_layer.iterkeys())
        #assert(test == list(range(n)))

        #here we create the linear expression in a very efficient manner. but a ordered list of gurobi variables is needed
        vars = [var for (key, var) in sorted(last_layer.iteritems())]
        for j in range(m):
            fun = LinExpr(weights[j, :], vars)
            fun.addConstant(biases[j])
            model.addLConstr(h[j] == fun, name=str(layer_num) + "c")

    return model, h
