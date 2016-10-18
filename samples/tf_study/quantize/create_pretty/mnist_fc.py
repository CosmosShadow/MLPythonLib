# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import cmtf.data.data_mnist as data_mnist

mnist = data_mnist.read_data_sets(one_hot=True)


# 输入
x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.float32, [None,10])

def create_model(images, phase):
	with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
		full_1 = images.fully_connected(10, name='full_1')
		full_1 = full_1.dropout(keep_prob=0.8, phase=phase)
		full_2 = full_1.fully_connected(10, name='full_2')
		return full_2

images = pt.wrap(x)
with tf.variable_scope('shakespeare'):
	last_full_train = create_model(images, pt.Phase.train)
with tf.variable_scope('shakespeare', reuse=True):
	last_full = create_model(images, pt.Phase.test)

output_eval = tf.identity(last_full, name='output')

output = tf.nn.softmax(last_full_train)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(output), [1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 正确率
correct_prediction = tf.equal(tf.argmax(output_eval,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.add_to_collection("x", x)
tf.add_to_collection("output", output_eval)

# run
with tf.Session() as sess:
	# init
	sess.run(tf.initialize_all_variables())
	# train
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		_, ac = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y: batch_ys})
		if i%100 == 0:
			print 'train: ', ac
	# test
	test_ac = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
	print test_ac

	saver = tf.train.Saver()
	tf.train.write_graph(sess.graph.as_graph_def(), './models', 'mnist_graph_def', as_text=False)
	saver.save(sess, "./models/mnist.ckpt")
















