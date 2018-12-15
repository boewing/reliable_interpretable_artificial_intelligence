
output_csv = open("exp4.csv", "w")
exp_num = 0
while True:
    fd = open("exp4/log_" + str(exp_num) + ".txt","r")
    for i, line in enumerate(fd):
        if i == 0:
            output_csv.write(line[:-1] + "; ")
            #print(line[:-1] + "; ")

        if line.__contains__("verified"):
            output_csv.write(line[:-1])
            #print(line[:-1])
        elif line.__contains__("can not be verified"):
            output_csv.write(line[:-1])
            #print(line[:-1])
        elif line.__contains__("image not correctly"):
            output_csv.write(line[:-1])
    output_csv.write("\n")
        #print(i, exp_num)

    fd.close()

    exp_num += 1
    if exp_num == 200:
        break

output_csv.close()