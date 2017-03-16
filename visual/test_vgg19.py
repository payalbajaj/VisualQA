import numpy as np
import tensorflow as tf
import vgg19
import utils
import glob

BATCH_SIZE = 10


filenames = glob.glob('/home/admin224N/VG_100K/*')
print len(filenames)
def print_visembed(nameimg, sumsecond, f):
	strtoadd = nameimg + " " + " ".join(str(i) for i in sumsecond) + "\n"
	f.write(strtoadd)  # python will convert \n to os.linesep

	
def imageget(startind, endind):
	imgs = []
	for i, filen in enumerate(filenames[startind:endind]):
		im = utils.load_image(filen)
		if len(im.shape) != 3:
			continue
		imgs.append(im.reshape((448, 448, 3)))
	imgsnp = np.array(imgs)
	return imgsnp

f = open('/home/admin224N/data_VisualQA/cnn.txt', 'w')
with tf.Session() as sess:
    size = len(filenames)
    for step in xrange(size / BATCH_SIZE):
      offset = step * BATCH_SIZE
      batch_data = imageget(offset,offset + BATCH_SIZE)
      images = tf.placeholder("float", [batch_data.shape[0], 448, 448, 3])
      feed_dict = {images: batch_data}
      vgg = vgg19.Vgg19()
      with tf.name_scope("content_vgg"):
      	vgg.build(images)
      output = sess.run(vgg.output, feed_dict=feed_dict)
      for ind, out in enumerate(output):
		npyout = np.array(out)
		sumit = npyout.reshape(-1)
		print sumit.shape
		print_visembed(filenames[offset+ind], sumit, f)
f.close()
