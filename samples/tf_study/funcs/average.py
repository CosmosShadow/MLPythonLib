# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)
ema = tf.train.ExponentialMovingAverage(decay=0.9)
ema_apply_op = ema.apply([x_plus_1])
ema_x = ema.average(x_plus_1)

with tf.control_dependencies([x_plus_1, ema_apply_op]):
    y = tf.identity(x)

with tf.Session() as session:
	session.run(tf.initialize_all_variables())
	for i in xrange(10):
		y.eval()
		print(ema_x.eval())