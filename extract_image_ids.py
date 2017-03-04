import json

data_path = "../data/question_answers.json"
image_ids = set()
# with open(data_path) as data_file:
# 	data = json.load(data_file)
# 	for indx in range(len(data)):
# 		img_iter = data[indx]
# 		# print img_iter
# 		img = str(img_iter["id"])
# 		for ques in img_iter["qas"]:
# 			if(" " not in ques["answer"]):
# 				image_ids.add(img)

# f = open("data/image_ids.txt", "w")
# for img in image_ids:
# 	f.write(img+"\n")
# f.close()

f = open("data/image_ids.txt", "r")
for line in f:
	image_ids.add(line.replace("\n",""))
f.close()

image_ids = list(image_ids)

import os
import shutil
idx = 0
for file_name in os.listdir("/media/pabajaj/Elements/VG_100K/"):
	image_id = file_name.split(".")[0]
	idx += 1
	if image_id in image_ids:
		print idx
		shutil.copy2(os.path.join("/media/pabajaj/Elements/VG_100K/", file_name), "/home/pabajaj/data_VisualQA/images/")
