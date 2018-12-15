from analyzer import *
import os

nets = ["6_200", "9_200", "4_1024"]

lookup = {}

for netname in nets:
    fd = open("../mnist_nets/mnist_relu_" + netname + ".txt","r")
    net = fd.read()
    net_adler = adler32(net)
    if net_adler not in lookup:
        lookup[net_adler] = {}
    else:
        print("Hash collision")
        exit(1)

    for i in range(100):
        fd = open("../mnist_images/img" + str(i) + ".txt","r")
        img = fd.read()
        img_adler = adler32(img)

        eps = 0.0
        if os.path.isfile(netname + "_boundary_epsilons/condormnist_relu_" + netname + "_img" + str(i) + ".out"):
            fd = open(netname + "_boundary_epsilons/condormnist_relu_" + netname + "_img" + str(i) + ".out", "r")
            for line in fd:
                if line[0] == "[":
                    eps = float(eval(line)[0])
                    assert (eps > 0.0)

        if eps == 0.0:
            print("No epsilon for img" + str(i) + " and net " + netname)
        else:
            if img_adler not in lookup[net_adler]:
                lookup[net_adler][img_adler] = eps
            else:
                print("Hash collision")
                exit(1)



print(lookup)