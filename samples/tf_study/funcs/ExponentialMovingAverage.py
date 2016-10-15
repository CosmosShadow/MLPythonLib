# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)	#自加1,并且值assign给新变量
ema = tf.train.ExponentialMovingAverage(decay=0.9)
ema_apply_op = ema.apply([x_plus_1])
ema_x = ema.average(x_plus_1)		#取应用后的内部变量，执行的时候不会使用op

with tf.control_dependencies([ema_apply_op]):
    y = tf.identity(x)

with tf.Session() as session:
	session.run(tf.initialize_all_variables())
	for i in xrange(10):
		# y.eval()
		# print(x.eval(), x_plus_1.eval(), ema_x.eval())
		print(session.run([y, ema_x]))
		# print(session.run(ema_x))












