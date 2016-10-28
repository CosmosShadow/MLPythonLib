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
		with pt.defaults_scope(activation_fn=tf.nn.relu):
			fc_seq = pt.wrap(x).sequential()
			fc_seq.fully_connected(10, activation_fn=None)
			fc_seq.softmax()
			return tf.nn.softmax(fc_seq)

def cnn():
	with tf.variable_scope('cnn'):
		with pt.defaults_scope(activation_fn=tf.nn.relu):
			x_reshape = tf.reshape(x, [-1, 28, 28, 1])
			cnn_seq = pt.wrap(x_reshape).sequential()
			cnn_seq.conv2d(7, 16)
			cnn_seq.max_pool(2, 2)
			cnn_seq.conv2d(7, 16)
			cnn_seq.max_pool(2, 2)
			cnn_seq.flatten()
			cnn_seq.fully_connected(32, activation_fn=tf.nn.relu)
			cnn_seq.fully_connected(10, activation_fn=None)
			return tf.nn.softmax(cnn_seq)

# ------------单fc-------------
# 平均时间: 1.07~1.32
# 正确率: 91.0%
# output = fc()

# ------------单cnn-------------
# 平均时间: 2.06~2.77
# 正确率: 98.8%
# output = cnn()

# ------------fc与cnn双网络-------------
# 平均时间: 1.27~2.19
# 正确率: 
output_fc = fc()
ratio = tf.reduce_max(output_fc) / tf.reduce_sum(output_fc)
def fc_identity():
	return tf.identity(output_fc)
output = tf.cond(tf.greater(ratio, 0.12), fc_identity, cnn)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(output), [1]))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
right_count_op = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(y,1)), tf.int32))

def evaluate():
	right_count = 0
	for _ in range(500):
		batch_xs, batch_ys = mnist.test.next_batch(1)
		right_count += sess.run(right_count_op, feed_dict={x: batch_xs, y: batch_ys})
	return right_count / 500.0

# GPU使用率
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6    #固定比例
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.initialize_all_variables())
	for epoch in range(100):
		start_time = time.time()
		for _ in range(500):
			batch_xs, batch_ys = mnist.train.next_batch(1)
			sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
		right_percent = evaluate()
		time_cost = time.time() - start_time
		print 'epoch: %2d   time: %.2f   right: %.1f%%' %(epoch, time_cost, right_percent*100.0)




