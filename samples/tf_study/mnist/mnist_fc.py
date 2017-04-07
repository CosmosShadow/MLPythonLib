# coding: utf-8
import tensorflow as tf
import cmtf.data.data_mnist as data_mnist

mnist = data_mnist.read_data_sets(one_hot=True)

#----------------模型----------------
# 输入
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])
# 模型
W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))
output = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(output), [1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 正确率
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# run
with tf.Session() as sess:
	# init
	sess.run(tf.global_variables_initializer())
	# train
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		_, ac = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y: batch_ys})
		if i%100 == 0:
			print 'train: ', ac
	# test
	test_ac = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
	print test_ac















