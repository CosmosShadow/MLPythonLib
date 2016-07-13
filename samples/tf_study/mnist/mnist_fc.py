# coding: utf-8
import tensorflow as tf
import cmtf.data.data_mnist as data_mnist

# 数据
mnist = data_mnist.read_data_sets(one_hot=True)

#----------------模型----------------
# 输入
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder("float", [None,10])
# 模型
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
output = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y*tf.log(output))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 正确率
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# run
with tf.Session() as sess:
	# init
	sess.run(tf.initialize_all_variables())
	# train
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
	# test
	print sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})