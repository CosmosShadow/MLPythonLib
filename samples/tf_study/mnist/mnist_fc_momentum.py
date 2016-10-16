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
cross_entropy = -tf.reduce_sum(y*tf.log(output))
batch = tf.Variable(0, dtype=tf.float32)
lr = tf.train.exponential_decay(0.001, batch*100, 100*10, 0.9, staircase=True)
train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(cross_entropy, global_step=batch)
# 正确率
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# run
with tf.Session() as sess:
	# init
	sess.run(tf.initialize_all_variables())
	# train
	for i in range(100):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		_, ac, lr_ = sess.run([train_step, accuracy, lr], feed_dict={x: batch_xs, y: batch_ys})
		if i%10 == 0:
			print 'train: %.2f, lr: %7.6f' % (ac, lr_)
	# test
	test_ac = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
	print '\ntest: ', test_ac















