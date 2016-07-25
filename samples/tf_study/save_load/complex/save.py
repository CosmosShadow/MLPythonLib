# coding: utf-8
import os
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist
import model

def save_model(saver, sess, step):
	task_dir = 'checkpoint'
	file_name = 'linear'
	if not os.path.exists(task_dir):
		os.makedirs(task_dir)
	saver.save(sess, os.path.join(task_dir, file_name), global_step = step)

timesteps = 10
x, y = model.create_model(timesteps)
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	x_val = np.ones([2, timesteps, 5])
	y_val = sess.run(y, feed_dict={x: x_val})
	print(y_val)

	save_model(saver, sess, 10)
	save_model(saver, sess, 20)

"""output
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
 [ 0.9186061 ]]
"""