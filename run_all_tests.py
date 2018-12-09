import os

os.system("echo This program runs all images and nets and return for each combination a file with the lower and the upper bound")

netlist = ["mnist_relu_3_10", "mnist_relu_3_20", "mnist_relu_3_50", "mnist_relu_4_1024", "mnist_relu_6_20", "mnist_relu_6_50", "mnist_relu_6_100", "mnist_relu_6_200", "mnist_relu_9_100", "mnist_relu_9_200"]
for netname in netlist:
	for i in range(100):
		os.system("bsub -W 48:00 -J " + netname +"_img" + str(i) + " -oo " + netname +"_img" + str(i) + ".log python run_tests.py ../mnist_nets/" + netname + ".txt ../mnist_images/img" + str(i) + ".txt")
