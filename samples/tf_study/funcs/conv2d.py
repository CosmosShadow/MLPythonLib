# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

kernal = 5
stride = 2
x = tf.Variable(tf.constant(0.0, shape=[1, 32, 32, 16]), name='x', trainable=True)
w = tf.get_variable('w', [kernal, kernal, 16, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
# conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='VALID')
with tf.control_dependencies([conv]):
	og_shape = conv.get_shape().as_list()
	print(og_shape)

with tf.Session() as session:
	session.run(tf.initialize_all_variables())
	session.run(conv)