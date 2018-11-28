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

    def add_affine(self,weights, biases):
        self.last_layer_num += 1
        self.model, self.last_layer = add_affine(self.model, weights, biases, self.last_layer, self.last_layer_num)

    def go_to_box(self):
        return go_to_box(self.model, self.last_layer)

    def verify_label(self):
        return verify_label(self.model, self.last_layer, self.label)


def verify_label(model,last_layer,label):
    #  this function is a performance improvement for the last layer bounds, it can prove that
    #  all other labels than label have a strictly smaller logit value than the label.
    #  can only be called for the last layer of the network
    #  it gives tighter bounds than the bounds comparison.

    #prepare the indices which are not the label
    r = list(range(len(last_layer)))
    del r[label]
    for i in r:
        model.setObjective(last_layer[i] - last_layer[label], GRB.MAXIMIZE)
        #model.setParam('BestBdStop', 0.0)
        #model.update()
        model.optimize()
        #print(model.Status)
        obj = model.getObjective()
        obv = obj.getValue()
        print("objective value of comparison: " + str(obv))
        #  if model.Status == GRB.USER_OBJ_LIMIT:
        if obv > 0:
            return False

    return True


def go_to_box(model, last_layer):
    n = len(last_layer)
    UB = np.zeros(n)
    LB = np.zeros(n)
    for i in range(n):
        model.setObjective(last_layer[i], GRB.MAXIMIZE)
        model.setParam('TimeLimit',.1)
        model.optimize()
        #   model.write("debug.lp")
        UB[i] = last_layer[i].x

        model.setObjective(last_layer[i], GRB.MINIMIZE)
        model.optimize()
        LB[i] = last_layer[i].x

    return LB, UB

def bounds_to_model(LB,UB,layer_num):
    model = Model("myLP")
    model.setParam('OutputFlag',False)
    last_layer = model.addVars(len(LB), name=str(layer_num)+"v", lb=LB, ub=UB)
    return model, last_layer


def add_ReLu(model, last_layer, layer_num):
    LB, UB = go_to_box(model, last_layer)

    n = len(last_layer)
    r = model.addVars(n, name=str(layer_num)+"v",lb=-GRB.INFINITY, ub=GRB.INFINITY)
    for k in range(n):
        #  ik = model.getVarByName(str(layer_num-1) + "v[" + str(k) + "]")
        if LB[k] >= 0:
            model.addLConstr(r[k] == last_layer[k], str(layer_num) + "c"+str(k))
        else:
            if UB[k] <=0:
                model.addLConstr(r[k] == 0, str(layer_num) + "c"+str(k))
            else:
                model.addLConstr(r[k] >= 0, str(layer_num) + "c"+str(k))
                model.addLConstr(r[k] >= last_layer[k], str(layer_num) + "cc"+str(k))
                lam = UB[k]/(UB[k]-LB[k])
                d = -LB[k]*lam
                model.addLConstr(r[k] <= lam*last_layer[k] + d, str(layer_num) + "ccc"+str(k))
    return model, r


def add_affine(model, weights, biases, last_layer, layer_num):
    size = weights.shape
    m = size[0]
    #  n = size[1]
    h = model.addVars(m, name=str(layer_num) + "v",lb=-GRB.INFINITY, ub=GRB.INFINITY)
    #  print({i: e for i, e in enumerate(weights[1, :])})
    #  print(last_layer)
    model.addConstrs((h[j] == biases[j] + last_layer.prod({i: e for i, e in enumerate(weights[j, :])}) for j in range(m)), name=str(layer_num) + "c")
    return model, h
