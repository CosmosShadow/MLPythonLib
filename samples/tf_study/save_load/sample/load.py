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

x, y = model.create_model()
saver = tf.train.Saver()

with tf.Session() as sess:
	# sess.run(tf.initialize_all_variables())
	load_model(saver, sess)
	x_val = np.ones([2, 10])
	y_val = sess.run(y, feed_dict={x: x_val})
	print(y_val)

"""output should be: 
[[-0.30422074]
 [-0.30422074]]
"""