# coding: utf-8
import time
import numpy as np
import tensorflow as tf
import cmtf.data.data_mnist as data_mnist

mnist = data_mnist.read_data_sets(one_hot=True)

#----------------模型----------------
# 输入
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder("float", [None,10])
keep_prob = tf.placeholder("float")

# 模型
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def conv(inp, kw, kh, ic, oc):
	W = weight_variable([kw, kh, ic, oc])
	b = bias_variable([oc])
	h = tf.nn.relu(conv2d(inp, W) + b)
	p = max_pool_2x2(h)
	return p
def fc(inp, input_size, output_size):
	W = weight_variable([input_size, output_size])
	b = bias_variable([output_size])
	outp = tf.matmul(inp, W) + b
	return outp

x_image = tf.reshape(x, [-1,28,28,1])
conv1 = conv(x_image, 5, 5, 1, 32)
conv2 = conv(conv1, 5, 5, 32, 64)
fc1 = fc(tf.reshape(conv2, [-1, 7*7*64]), 7*7*64, 1024)
fc1_relu = tf.nn.relu(fc1)
fc1_drop = tf.nn.dropout(fc1_relu, keep_prob)
fc2 = fc(fc1_drop, 1024, 10)
output=tf.nn.softmax(fc2)

cross_entropy = -tf.reduce_sum(y*tf.log(output))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# 测试
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for index in range(10):
		start = time.time()
		acc_arr = []
		for _ in range(100):
			batch_x, batch_y = mnist.train.next_batch(50)
			train_step.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
			acc = accuracy.eval(feed_dict={x:batch_x, y: batch_y, keep_prob: 1.0})
			acc_arr.append(acc)
		print "index: %2d   time: %.2f   acc: %.4f" % (index, time.time()-start, np.array(acc_arr).mean())
		

	batch_x, batch_y = mnist.test.next_batch(500)
	test_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
	print "test accuracy %g" % test_accuracy











