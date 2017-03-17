import random
import numpy as np
import other
import os

filenames = []
f = open('./VisualQA/data/image_ids.txt', 'r')
for line in f:
	filenames.append(line.strip())
f.close()

filenames = random.shuffle(filenames)

numfile = 0
imgs = []
for i, filen in enumerate(filenames):
	if not (os.path.exists("./VG_100K/"+filen+".jpg")):
		continue
	im = other.load_image("./VG_100K/" + filen + ".jpg")
	if len(im.shape) != 3:
		continue
	if numfile > 2000:
		break
	numfile += 1
	os.system("mv ./VG_100K/" + filen + ".jpg ./needed/")
