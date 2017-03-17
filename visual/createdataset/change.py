import numpy as np

f = open('./cnn.txt', 'r')
fileout = open('./baseline.txt', 'w')
for line in f:
	filid = line.split(" ")[0].split('/')[4].split('.')[0]
	vals = line.split(" ")[1:]
	vals = [float(i) for i in vals]
	vals = np.array(vals).reshape(196, 512)
	sumit = np.sum(vals, axis = 0)
	other = np.divide(sumit, 196).reshape(-1)
	otherw = [str(i) for i in other]
	fileout.write(filid + ' ' + ' '.join(otherw) + '\n')
