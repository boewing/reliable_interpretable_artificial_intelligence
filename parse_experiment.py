from datetime import datetime
from glob import glob
import os

class Exp:
	def __init__(self, num):
		self.error=""
		self.num = num

	def to_csv_line(self):
		
		delim=','
		line=str(self.num)+delim
		line+=self.cmd+delim
		for i in range(len(self.net)):
			line+=self.net[i]+delim
			line+=self.strategy[i]+delim
			try:
				line+=str((self.times[1+i]-self.times[i]).total_seconds())+delim
			except:
				line+="error"+delim

		if not self.error=="error":
			line+=delim+str(self.total_time)
		return line

	def parse(self, lines):

		#remove gurobi starter line
		lines =[x for x in lines if x != 'Academic license - for non-commercial use only']
	
		self.cmd = lines[0]
	
		#find strategy
		self.strategy = lines[1].replace('[','').replace(']','').replace(',','').replace('\'','').split()
		
		#find net structure
		self.net = lines[4].replace('[','').replace(']','').replace(',','').replace('\'','').split()
		
		time_lines = [3, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29]
		self.times =[]
		for l in time_lines:
			#t = datetime.strptime("Tue May 08 15:14:45 +0800 2012","%a %b %d %H:%M:%S %z %Y")
			if l>=len(lines):
				self.error="error"	
				return
			self.times.append(datetime.strptime(lines[l][5:],"%H:%M:%S.%f"))
		
		
		for l in range(30, len(lines)):
			if lines[l].startswith("time"):
				self.times.append(datetime.strptime(lines[l][5:],"%H:%M:%S.%f"))
			elif lines[l]=="verified" or lines[l]=="can not be verified":
				self.result = lines[l]
			elif lines[l].startswith("analysis"):
				self.total_time=float(lines[l][16:-9])

out=""
files = glob("/home/riai2018/riai/exp/*.txt")
files.sort()
#print(files)
for file in files:
	print(file)
	exp_num=os.path.basename(file)[3:-4]
	lines = open(file).read().splitlines()
	exp = Exp(exp_num)
	exp.parse(lines)
	out+=exp.to_csv_line()+'\n'
f = open("exp1_summary.csv", "w")
f.write(out)
f.close()
 
