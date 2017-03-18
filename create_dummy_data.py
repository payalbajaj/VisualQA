f_out = open("data/cnn.txt","w")
f = open("data/cnn_ip.txt","r")
for line in f:
	line_new = line.replace("\n","")+" 0.0"*512+"\n"
	f_out.write(line_new)
f.close()
f_out.close()
