import numpy as np

f = open('./T5.txt', 'r')
tracc, devacc = [], []
for line in f:
	tracc.append(line.strip().split(',')[3])
	devacc.append(line.strip().split(',')[5])
f.close
np.savetxt('./trainT5.txt', np.array(tracc), delimiter = '\n', fmt="%s")
np.savetxt('./devT5.txt', np.array(devacc), delimiter = '\n', fmt="%s")
