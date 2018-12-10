from noelina_analyzer import *
import numpy as np

def find_border_eps(netname, specname, strategy):
    e = 0.005
    e_l = 0.005
    e_u = 0.1
    while e_u - e_l > 0.001:
        v = verify(netname, specname, e, strategy)
        if v is True:
            e_l = e
            e += +(e_u - e)/2
        else:
            e_u = e
            e += -(e - e_l)/2
        print("epsilon is in:")
        print("[" +str(e_l) + ", " + str(e_u) + "]")

    return e


def verify(netname, specname, epsilon, strategy):
    # c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    #print("shape of net = " + str(nn.get_shape()))
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low, 0)

    label = int(x0_low[0])
    # own_label = get_label(nn, LB_N0)
    # label = own_label
    #  print("##############label", label, "own_label", own_label)
    start = time.time()
    verified_flag = False
    if (label == int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low, epsilon)
        _, verified_flag = analyze(nn, LB_N0, UB_N0, label, strategy)
        if (verified_flag):
            print("verified")
        else:
            print("can not be verified")
    else:
        print("image not correctly classified by the network. expected label ", int(x0_low[0]), " classified label: ",
              label)
    end = time.time()
    print("analysis time: ", (end - start), " seconds")
    return verified_flag

if __name__ == '__main__':
    from sys import argv
    netpath = argv[1]
    specpath = argv[2]

    #netname = netpath[netpath.find("mnist_relu"):netpath.find(".txt")]
    #specname = specpath[specpath.find("img"):specpath.find(".txt")]

    find_border_eps(netpath, specpath, ["LP"]*100)