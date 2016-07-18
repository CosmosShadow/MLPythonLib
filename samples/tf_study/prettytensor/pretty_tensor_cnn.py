# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

# 数据
mnist = data_mnist.read_data_sets(one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])

x_reshape = tf.reshape(x, [-1, 28, 28, 1])
seq = pt.wrap(x_reshape).sequential()
with pt.defaults_scope(activation_fn=tf.nn.relu):
	with seq.subdivide(2) as towers:
		towers[0].conv2d([3, 3], 16).max_pool(2, 2)
		towers[1].conv2d([3, 3], 32).max_pool(2, 2)
seq.flatten()
seq.fully_connected(64, activation_fn=tf.nn.relu)
seq.fully_connected(10, activation_fn=None)

softmax, loss = seq.softmax_classifier(10, labels=y)
accuracy = softmax.evaluate_classifier(y)
optimizer = tf.train.GradientDescentOptimizer(0.01)  # learning rate
train_op = pt.apply_optimizer(optimizer, losses=[loss])

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	# train
	for i in range(2000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		_, loss_val = sess.run([train_op, loss], feed_dict={x: batch_xs, y: batch_ys})
		if (i+1)%100 == 0:
			print 'index: %d, loss: %f' % (i+1, loss_val)
	# test
	accuracy_value = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
	print 'Accuracy: %g' % accuracy_value