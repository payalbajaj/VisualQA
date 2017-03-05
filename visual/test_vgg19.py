import numpy as np
import tensorflow as tf
import vgg19
import utils

filenames = []
f = open('../data/image_ids.txt', 'r')
for line in f:
	filenames.append(line.strip())
f.close()

def print_visembed(nameimg, sumsecond, f):
	strtoadd = nameimg + " " + " ".join([str(i) for i in sumsecond]) + "\n"
	f.write(strtoadd)  # python will convert \n to os.linesep

f = open('../cnn.txt', 'w')
with tf.Session() as sess:
	for i, filen in enumerate(filenames):
		imgs = []
		im = utils.load_image("/home/alisha/VG_100K/" + filen + ".jpg")
		if len(im.shape) != 3:
			continue
		imgs.append(im.reshape((224, 224, 3)))

		images = tf.placeholder("float", [1, 224, 224, 3])
        	feed_dict = {images: imgs}

        	vgg = vgg19.Vgg19()
        	with tf.name_scope("content_vgg"):
            		vgg.build(images)

        	output = sess.run(vgg.output, feed_dict=feed_dict)
        	for ind, out in enumerate(output):
			npyout = np.array(out)
			sumfirst = np.sum(npyout, axis = 0)
			sumsecond = np.sum(sumfirst, axis =0)
			print_visembed(filen, sumsecond, f)
f.close()
