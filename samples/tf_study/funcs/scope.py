# coding: utf-8
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
	with tf.variable_scope('foo'):
		with tf.variable_scope('bar'):
			v = tf.get_variable('v', [1])
	print 'v.name: ', v.name

	with tf.variable_scope('a', initializer=tf.constant_initializer(0.4)):
		v = tf.get_variable('v', [1])
		w = tf.get_variable('w', [1], initializer=tf.constant_initializer(0.3))
		sess.run(tf.initialize_all_variables())
		print 'e value: ', v.eval()
		print 'w value: ', w.eval()

	with tf.variable_scope('b') as b_scope:
		x = 1.0 + tf.get_variable('v', [1])
	print 'x.op.name: ', x.op.name
	print 'x.name: ', x.name

	with tf.variable_scope('c'):
		with tf.name_scope('d'):
			v = tf.get_variable('v', [1])
			x = 1.0 + v
	print 'v.name: ', v.name
	print 'x.name: ', x.name
	print 'x.op.name: ', x.op.name




