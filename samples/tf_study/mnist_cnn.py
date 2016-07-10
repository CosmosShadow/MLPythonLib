# coding: utf-8
import tensorflow as tf
import cmtf.data.data_mnist as data_mnist

mnist = data_mnist.read_data_sets(one_hot=True)

#----------------模型----------------
# 输入
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder("float", [None,10])

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
def fc(input, input_size, output_size):
	W_fc1 = weight_variable([input_size, output_size])
	b_fc1 = bias_variable([output_size])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	return h_fc1

x_image = tf.reshape(x, [-1,28,28,1])
conv1 = conv(x_image, 5, 5, 1, 32)
conv2 = conv(conv1, 5, 5, 32, 64)
h_pool2_flat = tf.reshape(conv2, [-1, 7*7*64])
h_fc1 = fc(h_pool2_flat, 7*7*64, 1024)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
output=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y*tf.log(output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 测试
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for i in range(50):
		batch_x, batch_y = mnist.train.next_batch(50)
		if i%10 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y, keep_prob: 1.0})
			print "step %d, training accuracy %g"%(i, train_accuracy)
		train_step.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

	print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})











