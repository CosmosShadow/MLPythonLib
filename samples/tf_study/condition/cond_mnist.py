# coding: utf-8
import time
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

# 数据
mnist = data_mnist.read_data_sets(one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])

def fc():
	with tf.variable_scope('fc'):
		fc_seq = pt.wrap(x).sequential()
		fc_seq.fully_connected(10, activation_fn=None)
		fc_seq.softmax()
		return tf.nn.softmax(fc_seq)

def cnn():
	with tf.variable_scope('cnn'):
		x_reshape = tf.reshape(x, [-1, 28, 28, 1])
		cnn_seq = pt.wrap(x_reshape).sequential()
		cnn_seq.conv2d(3, 16)
		cnn_seq.conv2d(3, 16)
		cnn_seq.conv2d(3, 16)
		cnn_seq.conv2d(3, 16)
		cnn_seq.max_pool(2, 2)
		cnn_seq.conv2d(7, 16)
		cnn_seq.max_pool(2, 2)
		cnn_seq.flatten()
		cnn_seq.fully_connected(32, activation_fn=tf.nn.relu)
		cnn_seq.fully_connected(10, activation_fn=None)
		return tf.nn.softmax(cnn_seq)

# ------------单fc-------------
# 平均时间: 1.07~1.32
# output = fc()

# ------------单cnn-------------
# 平均时间: 3.43~3.51
# output = cnn()

# ------------fc与cnn双网络-------------
# 平均时间: 4.11~4.69
output_fc = fc()
ratio = tf.reduce_max(output_fc) / tf.reduce_sum(output_fc)
def fc_identity():
	return tf.identity(output_fc)
output = tf.cond(tf.greater(ratio, 0.1), fc_identity, cnn)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(output), [1]))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
right_count_op = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(y,1)), tf.int32))

def evaluate():
	right_count = 0
	for _ in range(500):
		batch_xs, batch_ys = mnist.test.next_batch(1)
		right_count += sess.run(right_count_op, feed_dict={x: batch_xs, y: batch_ys})
	return right_count / 500.0

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for _ in range(20):
		start_time = time.time()
		for _ in range(500):
			batch_xs, batch_ys = mnist.train.next_batch(1)
			sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
		right_percent = evaluate()
		time_cost = time.time() - start_time
		print 'time: %.2f   right: %.3f' %(time_cost, right_percent)




