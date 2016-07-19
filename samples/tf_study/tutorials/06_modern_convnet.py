"""Tutorial on how to build a convnet w/ modern changes, e.g.
Batch Normalization, Leaky rectifiers, and strided convolution.

Parag K. Mital, Jan 2016.
"""
# %%
import tensorflow as tf
from libs.batch_norm import batch_norm
from libs.activations import lrelu
from libs.connections import conv2d, linear
import cmtf.data.data_mnist as data_mnist

mnist = data_mnist.read_data_sets(one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool, name='is_training')
x_tensor = tf.reshape(x, [-1, 28, 28, 1])
h_1 = lrelu(batch_norm(conv2d(x_tensor, 32, name='conv1'), is_training, scope='bn1'), name='lrelu1')
h_2 = lrelu(batch_norm(conv2d(h_1, 64, name='conv2'), is_training, scope='bn2'), name='lrelu2')
h_3 = lrelu(batch_norm(conv2d(h_2, 64, name='conv3'), is_training, scope='bn3'), name='lrelu3')
h_3_flat = tf.reshape(h_3, [-1, 64 * 4 * 4])
h_4 = linear(h_3_flat, 10)
y_pred = tf.nn.softmax(h_4)

# %% Define loss/eval/training functions
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# %% We'll train in minibatches and report accuracy:
n_epochs = 10
batch_size = 100
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, is_training: True})
    print(sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, is_training: False}))













