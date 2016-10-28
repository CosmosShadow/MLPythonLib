# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.constant(0)
y = tf.constant(-1)
data = tf.constant(10)

def f1():
	z = tf.add(x, 1)
	return tf.Print(z, [data], message='---------------------f1 process------------------')

def f2():
	z = tf.add(x, 2)
	return tf.Print(z, [data], message='---------------------f2 process------------------')

r = tf.cond(tf.less(x, y), f1, f2)

condition = tf.placeholder(tf.int32, shape=[], name="condition")

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print sess.run(r)














