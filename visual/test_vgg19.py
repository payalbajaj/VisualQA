import numpy as np
import tensorflow as tf
import vgg19
import utils

BATCH_SIZE = 50


filenames = []
f = open('../data/image_ids.txt', 'r')
for line in f:
	filenames.append(line.strip())
f.close()

def print_visembed(nameimg, sumsecond, f):
	strtoadd = nameimg + " " + " ".join([str(i) for i in sumsecond]) + "\n"
	f.write(strtoadd)  # python will convert \n to os.linesep


imgs = []
f = open('./notused.txt', 'w')
for i, filen in enumerate(filenames):
		im = utils.load_image("/home/alisha/VG_100K/" + filen + ".jpg")
		if len(im.shape) != 3:
			f.write(filen + '\n')
			continue
		imgs.append(im.reshape((224, 224, 3)))
f.close()

imgsnp = np.array(imgs)
f = open('../cnn.txt', 'w')
with tf.Session() as sess:
    size = len(imgs)
    for step in xrange(size / BATCH_SIZE):
      offset = step * BATCH_SIZE
      batch_data = imgsnp[offset:(offset + BATCH_SIZE), :, :, :]
      images = tf.placeholder("float", [50, 224, 224, 3])
      feed_dict = {images: batch_data}
      vgg = vgg19.Vgg19()
      with tf.name_scope("content_vgg"):
      	vgg.build(images)
      output = sess.run(vgg.output, feed_dict=feed_dict)
      for ind, out in enumerate(output):
	npyout = np.array(out)
	sumit = np.sum(np.sum(npyout, axis = 0), axis = 0)
	print_visembed(filenames[offset+ind], sumit, f)
f.close()
