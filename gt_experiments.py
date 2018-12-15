import glob
import os

nets = ["../mnist_nets/mnist_relu_6_200.txt", "../mnist_nets/mnist_relu_9_200.txt"]
#nets = ["../mnist_nets/mnist_relu_6_50.txt", "../mnist_nets/mnist_relu_6_200.txt", "../mnist_nets/mnist_relu_9_200.txt"]
images = glob.glob("../mnist_images/img*.txt")
epsilon = 0.01
#cmd_first_part = "timeout 60 python3.6 analyzer.py " #8 min = 480
cmd_first_part = "timeout 480 python3 analyzer.py" #8 min = 480

exp_num=0
for net in nets:
	for img in images:
		cmd = cmd_first_part+" "+net+" "+img+" "+str(epsilon)
		print (cmd)
		os.system("echo "+cmd+"> exp4/log_"+str(exp_num)+".txt")
		os.system(cmd+">> exp4/log_"+str(exp_num)+".txt")
		exp_num+=1