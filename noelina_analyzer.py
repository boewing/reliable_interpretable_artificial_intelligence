import numpy as np
from io import StringIO
import re
import csv
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time
import datetime
from linear_solver_threaded import *

def analyze(nn, LB_N0, UB_N0, label, *args):

    start = time.time()
    nn.ffn_counter = 0
    numlayer = nn.numlayer

    print("time", datetime.datetime.now().time())
    myLP = net_in_LP(LB_N0, UB_N0, 0, label, start)

    strategyno = 0
    timeout = False
    for layerno in range(numlayer):
        if (nn.layertypes[layerno] in ['ReLU', 'Affine']):

            print("add affine layer to problem", strategyno, "strategy LP")
            print("time", datetime.datetime.now().time())
            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]
            myLP.add_affine(weights, biases)

            strategyno += 1

            # handle ReLU layer
            if (nn.layertypes[layerno] == 'ReLU'):
                print("add relu layer to problem", strategyno, "strategy LP")
                print("time", datetime.datetime.now().time())
                num_out_pixels = len(weights)
                myLP.add_ReLu()
                strategyno += 1
                #print("adding ReLu took " + str(time.time() - t) + "seconds")

            nn.ffn_counter += 1

        else:
            print(' net type not supported')

        # this works! So for each layer we go from our bound to alina and back
        # if we stay in the interval domain, we prob shouldn't go back and forth
        # LB_temp, UB_temp = alina_interval_to_bounds(man, element)
        # element = bounds_to_elina_interval(man, LB_temp, UB_temp)

    print("time", datetime.datetime.now().time())

    verified_flag = myLP.verify_label()
    return None, verified_flag


def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon

    for i in range(num_pixels):
        if (LB_N0[i] < 0):
            LB_N0[i] = 0
        if (UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i + 1])
            b = parse_bias(lines[i + 2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer += 1
            i += 3
        else:
            raise Exception('parse error: ' + lines[i])
    return res


def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    stringhandle = StringIO(str(text))
    #with open('dummy', 'w') as my_file:
    #    my_file.write(text)
    data = np.loadtxt(stringhandle, delimiter=',', dtype=np.double)
    low = np.copy(data[:, 0])
    high = np.copy(data[:, 1])
    return low, high

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
    # return v.reshape((v.size,1))
    return v


def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size, 1))
    # return v


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
            start = i + 1
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

if __name__ == '__main__':
    from sys import argv

    # if len(argv) < 3 or len(argv) > 4:
    #    print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
    #    exit(1)

    print(" LP"*18)
    print(" ")
    print(" ")

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    # c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    print("shape of net = " + str(nn.get_shape()))
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low, 0)

    label = int(x0_low[0])
    #own_label = get_label(nn, LB_N0)
    #label = own_label
    #  print("##############label", label, "own_label", own_label)
    start = time.time()
    if (label == int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low, epsilon)
        _, verified_flag = analyze(nn, LB_N0, UB_N0, label)
        if (verified_flag):
            print("verified")
        else:
            print("can not be verified")
    else:
        print("image not correctly classified by the network. expected label ", int(x0_low[0]), " classified label: ",
              label)
    end = time.time()
    print("analysis time: ", (end - start), " seconds")
