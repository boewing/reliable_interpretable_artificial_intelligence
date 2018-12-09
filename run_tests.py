import os
from glob import glob
import parse_experiment as pe

EXP3_NUM = 1

def run_binary_epsilon_search(lower_epsilon, upper_epsilon, cmd_first_part, search_iter, netname, specname):
	global EXP3_NUM
	if search_iter>10:
		return lower_epsilon, upper_epsilon

	epsilon = (lower_epsilon+upper_epsilon)/2
	strategy = " LP"*18
	cmd = cmd_first_part+ " "+str(epsilon) + strategy
	exp_num = EXP3_NUM
	EXP3_NUM+=1
	
	os.system("echo "+cmd+"> logs/log_"+netname +"_"+specname+ "_"+str(exp_num)+".txt")
	os.system(cmd+">> logs/log_"+netname +"_"+specname+ "_"+str(exp_num)+".txt")

	lines = open("logs/log_"+netname +"_"+specname+ "_"+str(exp_num)+".txt").read().splitlines()
	exp = pe.Exp(exp_num)
	exp.parse(lines)

	if exp.result == "verified":
		lower_epsilon=epsilon
	else:
		upper_epsilon=epsilon

	return run_binary_epsilon_search(lower_epsilon, upper_epsilon, cmd_first_part, search_iter+1, netname, specname)

def run_and_print_search(lower_epsilon, upper_epsilon, cmd_first_part, netname, specname):
	lower_epsilon, upper_epsilon = run_binary_epsilon_search(lower_epsilon, upper_epsilon, cmd_first_part, 0, netname, specname)
	print ("########",cmd_first_part, "      ")
	print ("lower epsilon",lower_epsilon,"upper epsilon", upper_epsilon)


if __name__ == '__main__':
	from sys import argv
	global_lower_epsilon = 0.005
	global_upper_epsilon = 0.1
	netpath = argv[1]
	specpath = argv[2]
	cmd_first_part = "python3.6 analyzer.py " + netpath + " " + specpath

	netname = netpath[netpath.find("mnist_relu"):netpath.find(".txt")]
	specname = specpath[specpath.find("img"):specpath.find(".txt")]


	## all LP

	run_and_print_search(global_lower_epsilon, global_upper_epsilon, cmd_first_part, netname, specname)