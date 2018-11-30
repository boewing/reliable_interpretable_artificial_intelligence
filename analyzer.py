import sys
sys.path.insert(0, '../ELINA/python_interface/')


import numpy as np
import re
import csv
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time
import datetime

from linear_solver import *

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')

STRAT = None

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0

    def get_shape(self):
        res = []
        for i in range(self.numlayer):
            res.append(len(self.biases[i]))
        return res

def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    #return v.reshape((v.size,1))
    return v

def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size,1))
    #return v

def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result

def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])

def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i+1])
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer+= 1
            i += 3
        else:
            raise Exception('parse error: '+lines[i])
    return res
   
def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',',dtype=np.double)
    low = np.copy(data[:,0])
    high = np.copy(data[:,1])
    return low,high

def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon
     
    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0,i,weights[i])
    return linexpr0

def affine_box_layerwise(man,element,weights, biases):

    """
    Performs the Affine operation
    
    Parameters
    ----------
    man : ElinaManagerPtr
        Pointer to the ElinaManager.
    elem : ElinaAbstract0Ptr
        Pointer to the ElinaAbstract0 which dimensions need to be assigned.
    weights : np Array
        The weights array.
    biases : np Array
        The biases array

    Returns
    -------
    res : ElinaAbstract0Ptr
        Pointer to the new abstract object.

    """

    dims = elina_abstract0_dimension(man,element)
    num_in_pixels = dims.intdim + dims.realdim
    num_out_pixels = len(weights)
 
    dimadd = elina_dimchange_alloc(0,num_out_pixels)    
    for i in range(num_out_pixels):
        dimadd.contents.dim[i] = num_in_pixels
    elina_abstract0_add_dimensions(man, True, element, dimadd, False)
    elina_dimchange_free(dimadd)
    np.ascontiguousarray(weights, dtype=np.double)
    np.ascontiguousarray(biases, dtype=np.double)
    var = num_in_pixels
    # handle affine layer
    for i in range(num_out_pixels):
        tdim= ElinaDim(var)
        linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
        element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
        var+=1
    dimrem = elina_dimchange_alloc(0,num_in_pixels)
    for i in range(num_in_pixels):
        dimrem.contents.dim[i] = i
    elina_abstract0_remove_dimensions(man, True, element, dimrem)
    elina_dimchange_free(dimrem)
    return element

def bounds_to_elina_interval(man, LB, UB):
    num_pixels = len(LB)
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB[i],UB[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)
    return element

def alina_interval_to_bounds(man, element):
    LB = []
    UB = []
    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)
    for i in range(output_size):
        LB.append(bounds[i].contents.inf.contents.val.dbl)
        UB.append(bounds[i].contents.sup.contents.val.dbl)
    elina_interval_array_free(bounds,output_size)
    return LB, UB

def get_label(nn, input_img):
    # this function does one forward path through the entire net and labels the input
    LB_N0 = input_img
    UB_N0 = input_img
    nn.ffn_counter = 0
    numlayer = nn.numlayer 
    man = elina_box_manager_alloc()
    element = bounds_to_elina_interval(man, LB_N0, UB_N0)

    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter]
           element = affine_box_layerwise(man,element,weights, biases)

           # handle ReLU layer 
           if(nn.layertypes[layerno]=='ReLU'):
              num_out_pixels = len(weights)
              element = relu_box_layerwise(man,True,element,0, num_out_pixels)
           nn.ffn_counter+=1 

        else:
           print(' net type not supported')
   
    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    final_LB, final_UB = alina_interval_to_bounds(man, element)
    elina_abstract0_free(man, element)
    elina_manager_free(man)

    # try to classify
    predicted_label = 0
    
    for i in range(output_size):
        inf = final_LB[i] #bounds[i].contents.inf.contents.val.dbl
        flag = True
        for j in range(output_size):
            if(j!=i):
               sup = final_UB[j] #bounds[j].contents.sup.contents.val.dbl
               if(inf<=sup):
                  flag = False
                  break
        if(flag):
            predicted_label = i
            break    
    return predicted_label

class Oracle:
    def __init__(self, nn):
        self.nn = nn
        self.layer_types=[]
        self.nn_layer_link =[]
        for num,nn_layer in enumerate(nn.layertypes):
            if nn_layer=='ReLU':
                self.layer_types.append('affine')
                self.layer_types.append('relu')
                self.nn_layer_link.append(num)
                self.nn_layer_link.append(num)
            elif nn_layer=='Affine':
                self.layer_types.append('affine')
                self.nn_layer_link.append(num)
        #print(nn.layertypes)
        #print(nn.numlayer)
        print(self.layer_types)
        #print(self.nn_layer_link)
    
    def get_strategy(self):

        a = 0
        b = 0
        temp = ['box']*a + ['LP'] * (len(self.layer_types) -b - a) + ['box']*b

        return temp


def analyze(nn, LB_N0, UB_N0, label):   
    nn.ffn_counter = 0
    numlayer = nn.numlayer 
    man = elina_box_manager_alloc()
    
    print ("time",datetime.datetime.now().time())
    oracle = Oracle(nn)
    #strategy = oracle.get_strategy()
    if len(STRAT) == 0:
        strategy = oracle.get_strategy()
    else:
        strategy = STRAT
    #print (strategy)
    if strategy[0]=='box':
        element = bounds_to_elina_interval(man, LB_N0, UB_N0)
        myLP = None
    elif strategy[0]=='LP':
        element = None
        myLP = net_in_LP(LB_N0, UB_N0, 0, label)

    strategyno=0
    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           print ("add affine layer to problem", strategyno, "strategy",strategy[strategyno])
           print ("time",datetime.datetime.now().time())
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter]
           if strategy[strategyno]=='box':
                if element==None: #we come from LP and go to box
                    LB, UB = myLP.go_to_box()
                    element = bounds_to_elina_interval(man, LB, UB)
                    element = affine_box_layerwise(man,element,weights, biases)
                    myLP = None
                elif myLP==None: #we stay in box
                    element = affine_box_layerwise(man,element,weights, biases)
           elif strategy[strategyno]=='LP':
                if element==None: #we stay in LP
                    myLP.add_affine(weights,biases)
                elif myLP==None: #we come from box and go to LP
                    LB, UB = alina_interval_to_bounds(man, element)
                    myLP = net_in_LP(LB, UB, 0, label)
                    myLP.add_affine(weights,biases)
                    element = None
           else:
                print ("not valid strategy", strategy[strategyno])
                exit(0)
           strategyno+=1
           #  Question: is it necessary to increase the strategyno twice?

           # handle ReLU layer 
           if(nn.layertypes[layerno]=='ReLU'):
                print ("add relu layer to problem", strategyno, "strategy",strategy[strategyno])
                print ("time",datetime.datetime.now().time())
                num_out_pixels = len(weights)
                if strategy[strategyno]=='box':
                    if element==None: #we come from LP and go to box
                          LB, UB = myLP.go_to_box()
                          element = bounds_to_elina_interval(man, LB, UB)
                          element = relu_box_layerwise(man,True,element,0, num_out_pixels)
                          myLP = None
                    elif myLP==None: #we stay in box
                          element = relu_box_layerwise(man,True,element,0, num_out_pixels)
                elif strategy[strategyno]=='LP':
                    if element==None: #we stay in LP
                          myLP.add_ReLu()
                    elif myLP==None: #we come from box and go to LP
                          LB, UB = alina_interval_to_bounds(man, element)
                          myLP = net_in_LP(LB, UB, 0, label)
                          myLP.add_ReLu()
                          element = None
                else:
                    print ("not valid strategy", strategy[strategyno])
                    exit(0)
                strategyno+=1

           nn.ffn_counter+=1 

        else:
           print(' net type not supported')

        # this works! So for each layer we go from our bound to alina and back
        # if we stay in the interval domain, we prob shouldn't go back and forth
        #LB_temp, UB_temp = alina_interval_to_bounds(man, element)
        #element = bounds_to_elina_interval(man, LB_temp, UB_temp)

    print ("time",datetime.datetime.now().time())
    # get bounds for each output neuron
    if myLP==None:
        final_LB, final_UB = alina_interval_to_bounds(man, element)
        output_size = len(final_LB)
        # print upper and lower bounds for debug
        for i in range(output_size):
            print("converted neuron", i, "lower bound", final_LB[i], "upper bound", final_UB[i])
            # print("LP neuron", i, "lower bound", LP_LB[i], "upper bound", LP_UB[i])

        # if epsilon is zero, try to classify else verify robustness

        verified_flag = True
        predicted_label = 0
        if (LB_N0[0] == UB_N0[0]):
            for i in range(output_size):
                inf = final_LB[i]  # bounds[i].contents.inf.contents.val.dbl
                flag = True
                for j in range(output_size):
                    if (j != i):
                        sup = final_UB[j]  # bounds[j].contents.sup.contents.val.dbl
                        if (inf <= sup):
                            flag = False
                            break
                if (flag):
                    predicted_label = i
                    break
        else:
            # for a label to be verified, all upper bounds of the intervals have to be below (<=) the lower bound of the the label interval
            # inf = bounds[label].contents.inf.contents.val.dbl #inf is the lower bound of an interval
            inf = final_LB[label]
            for j in range(output_size):
                if (j != label):
                    # sup = bounds[j].contents.sup.contents.val.dbl #sup is the upper bound of an interval
                    sup = final_UB[j]
                    if (inf <= sup):
                        predicted_label = label
                        verified_flag = False
                        break
        #dims = elina_abstract0_dimension(man, element)
        #output_size = dims.intdim + dims.realdim
    elif element==None:
        #final_LB, final_UB = myLP.go_to_box()
        verified_flag = myLP.verify_label()
        predicted_label = label

    #elina_interval_array_free(bounds,output_size)
    if element!=None:
        elina_abstract0_free(man,element)
    elina_manager_free(man)
    print ("time",datetime.datetime.now().time())
    return predicted_label, verified_flag



if __name__ == '__main__':
    from sys import argv
    #if len(argv) < 3 or len(argv) > 4:
    #    print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
    #    exit(1)

    #m = Model()
    #h = m.addVars(2,lb=[-1, -2], ub=[2, 3])
    #m.write("debug.lp")
    #add_ReLu(m,h,0)
    #add_affine(m,np.array([[1 ,2],[3,4]]), np.array([5, 6]),h,0)
    #m.write("debug_end.lp")

    STRAT = [argv[i] for i in range(4,len(argv))]
    print (STRAT)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    #c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    print("shape of net = " + str(nn.get_shape()))
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)
    
    #  label, _ = analyze(nn,LB_N0,UB_N0,0)
    own_label = get_label(nn, LB_N0)
    label = own_label
    #  print("##############label", label, "own_label", own_label)
    if label != own_label:
        exit(0)
    start = time.time()
    if(label==int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag = analyze(nn,LB_N0,UB_N0,label)
        if(verified_flag):
            print("verified")
        else:
            print("can not be verified")  
    else:
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")
    

