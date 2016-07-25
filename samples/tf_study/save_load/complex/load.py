# coding: utf-8
import os
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist
import model

def load_model(saver, sess):
	checkpoint_dir = 'checkpoint'
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
	else:
		raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

timesteps = 20
x, y = model.create_model(timesteps)
saver = tf.train.Saver()

with tf.Session() as sess:
	# sess.run(tf.initialize_all_variables())
	load_model(saver, sess)
	x_val = np.ones([2, timesteps, 5])
	y_val = sess.run(y, feed_dict={x: x_val})
	print(y_val)

"""output: 
[[ 0.18480827]
 [ 0.18480827]
 [ 0.42967057]
 [ 0.42967057]
 [ 0.65803349]
 [ 0.65803349]
 [ 0.79266781]
 [ 0.79266781]
 [ 0.85233486]
 [ 0.85233486]
 [ 0.88056421]
 [ 0.88056421]
 [ 0.89656115]
 [ 0.89656115]
 [ 0.90677094]
 [ 0.90677094]
 [ 0.91370642]
 [ 0.91370642]
 [ 0.9186061 ]
 [ 0.9186061 ]
 [ 0.92220283]
 [ 0.92220283]
 [ 0.92494094]
 [ 0.92494094]
 [ 0.92710233]
 [ 0.92710233]
 [ 0.92886484]
 [ 0.92886484]
 [ 0.93034244]
 [ 0.93034244]
 [ 0.93160754]
 [ 0.93160754]
 [ 0.93270683]
 [ 0.93270683]
 [ 0.93366969]
 [ 0.93366969]
 [ 0.93451595]
 [ 0.93451595]
 [ 0.93525869]
 [ 0.93525869]]
"""