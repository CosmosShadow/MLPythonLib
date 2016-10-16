# coding: utf-8
import tensorflow as tf
import cmtf.data.data_mnist as data_mnist

mnist = data_mnist.read_data_sets(one_hot=True)
sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), [1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
tf.initialize_all_variables().run()
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # train_step.run({x: batch_xs, y_: batch_ys})
  _, ac = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
  if i%100 == 0:
  	print 'train: ', ac
  # train_step.run({x: batch_xs, y_: batch_ys})

test_ac = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
print 'test: ', test_ac
