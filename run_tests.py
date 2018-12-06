import os
from glob import glob
import parse_experiment as pe

EXP3_NUM = 56

def run_binary_epsilon_search(lower_epsilon, upper_epsilon, cmd_first_part, strategy, search_iter):
	global EXP3_NUM
	if search_iter>10:
		return lower_epsilon, upper_epsilon

	epsilon = (lower_epsilon+upper_epsilon)/2
	cmd = cmd_first_part+str(epsilon)+strategy
	exp_num = EXP3_NUM
	EXP3_NUM+=1
	
	os.system("echo "+cmd+"> exp3/log"+str(exp_num)+".txt")
	os.system(cmd+">> exp3/log"+str(exp_num)+".txt")

	lines = open("exp3/log"+str(exp_num)+".txt").read().splitlines()
	exp = pe.Exp(exp_num)
	exp.parse(lines)

	if exp.result == "verified":
		lower_epsilon=epsilon
	else:
		upper_epsilon=epsilon

	return run_binary_epsilon_search(lower_epsilon, upper_epsilon, cmd_first_part, strategy, search_iter+1)

def run_and_print_search(lower_epsilon, upper_epsilon, cmd_first_part, strategy):
	lower_epsilon, upper_epsilon = run_binary_epsilon_search(lower_epsilon, upper_epsilon, cmd_first_part, strategy, 0)
	print ("########",cmd_first_part, "      ", strategy)
	print ("lower epsilon",lower_epsilon,"upper epsilon", upper_epsilon)


### exp 2
#strategy = " LP"*18
#all_nets = glob("/home/riai2018/mnist_nets/*.txt")[2:]
#print (all_nets)
#min_epsilon = 0.005
#max_epsilon = 0.1
#epsilon_steps = 8
#epsilons = [min_epsilon+i*(max_epsilon-min_epsilon)/(epsilon_steps-1) for i in range(epsilon_steps)]
#print (epsilons)
#
#exp_num=16
#for net in all_nets:
	#for epsilon in epsilons:
		#cmd = "python3 analyzer.py "+net+" ../mnist_images/img23.txt "+str(epsilon)+strategy
		#print (cmd)
		#os.system("echo "+cmd+"> exp2/log"+str(exp_num)+".txt")
		#os.system(cmd+">> exp2/log"+str(exp_num)+".txt")
		#exp_num+=1
#



### exp 3

global_lower_epsilon = 0.005
global_upper_epsilon = 0.1
cmd_first_part = "python3 analyzer.py ../mnist_nets/mnist_relu_9_200.txt ../mnist_images/img23.txt "
## all box vs LP at the end
strategy = " box"*18
#run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " box"*17+" LP"
#run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " box"*16+" LP LP"
#run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " box"*14+" LP LP LP LP"
#run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)

## all LP vs box in the middle
strategy = " LP"*18
#run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " LP"*9+" box"+" LP"*8
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " LP"*5+" box"+" LP"*5+" box"+" LP"*6
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " LP"*3+" box"+" LP"*5+" box"+" LP"*3+" box"+" LP"*4
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " LP"*3+" box"+" LP"*3+" box"+" LP"*3+" box"+" LP"*3+" box"+" LP"*2
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)


cmd_first_part = "python3 analyzer.py ../mnist_nets/mnist_relu_4_1024.txt ../mnist_images/img23.txt "
## all box vs LP at the end
strategy = " box"*7
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " box"*6+" LP"
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " box"*5+" LP LP"
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " box"*3+" LP LP LP LP"
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)

## all LP vs box in the middle
strategy = " LP"*7
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)
strategy = " LP"*3+" box"+" LP"*3
run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, strategy)

#cmd = cmd_first_part+str(0.005)+strategy
#exp_num = EXP3_NUM
#EXP3_NUM+=1
	
#os.system("echo "+cmd+"> exp3/"+str(exp_num)+".log")
#os.system(cmd+">> exp3/"+str(exp_num)+".log")


exit(0)
#cmd = "python3 analyzer.py ../mnist_nets/mnist_relu_6_50.txt ../mnist_images/img23.txt 0.1 box box box box box box box box box box box box"
