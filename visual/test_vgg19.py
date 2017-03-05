import numpy as np
import tensorflow as tf
import vgg19
import utils

namesofimgs = ["7", "6"]
img1 = utils.load_image("/home/alisha/VG_100K/7.jpg")
img2 = utils.load_image("/home/alisha/VG_100K/6.jpg")
print img1.shape
batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

def print_visembed(nameimg, sumsecond, f):
	strtoadd = nameimg + " " + " ".join([str(i) for i in sumsecond]) + "\n"
	f.write(strtoadd)  # python will convert \n to os.linesep
# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [2, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        output = sess.run(vgg.output, feed_dict=feed_dict)
        f = open('../cnn.txt', 'w')
	for ind, out in enumerate(output):
		npyout = np.array(out)
		sumfirst = np.sum(npyout, axis = 0)
		sumsecond = np.sum(sumfirst, axis =0)
		print_visembed(namesofimgs[ind], sumsecond, f)
	f.close()
