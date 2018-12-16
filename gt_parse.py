import glob
import os
import parse_experiment as pe

results=[]
for exp_num in range(100,200):
	lines = open("exp4/log_"+str(exp_num)+".txt").read().splitlines()
	exp = pe.Exp(exp_num)
	exp.parse(lines)
	relus_str=""
	for r in range(6):
		if r<len(exp.relus):
			relus_str+=";"+exp.relus[r]
		else:
			relus_str+=";"
	if hasattr(exp,'total_time'):
		total_time = ";"+str(exp.total_time)
	else:
		total_time = ";"
	if exp.result == "verified":
		results.append(exp.cmd+relus_str+total_time+";verified")
	elif exp.result == "can not be verified":
		results.append(exp.cmd+relus_str+total_time+";failed cannot")
	else:
		results.append(exp.cmd+relus_str+total_time+";failed timeout")

with open("exp4_9_200.csv",'w') as f:
	for exp_num in range(100):
		f.write(results[exp_num]+"\n")
