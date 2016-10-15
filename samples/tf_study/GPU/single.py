# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

# Creates a graph.
# with tf.device('/cpu:0'):
with tf.device('/gpu:0'):
	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
	c = tf.matmul(a, b)

config = tf.ConfigProto()
config.log_device_placement = True
# config.gpu_options.allow_growth = True		#动态增长
config.gpu_options.per_process_gpu_memory_fraction = 0.4		#固定比例
config.allow_soft_placement = True		#如果device指定出错，则可以动态切换到其它的去，而不报错

sess = tf.Session(config=config)
# Runs the op.
print sess.run(c)