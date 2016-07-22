# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
import prettytensor as pt
from prettytensor.tutorial import data_utils

tf.app.flags.DEFINE_string('save_path', None, 'Where to save the model checkpoints on local disk. Checkpoints are in LevelDb.')
FLAGS = tf.app.flags.FLAGS


BATCH_SIZE = 32
CHARS = 128
TIMESTEPS = 15
SEXES = 2
EMBEDDING_SIZE = 16


def create_model(text_in, labels, timesteps, per_example_weights, phase=pt.Phase.train):
  with pt.defaults_scope(phase=phase, l2loss=0.00001):
    with tf.device('/cpu:0'):
      embedded = text_in.embedding_lookup(CHARS, [EMBEDDING_SIZE])
    # We need to cleave the sequence because sequence lstm expect each
    # timestep to be in its own Tensor.
    lstm = (embedded.cleave_sequence(timesteps).sequence_lstm(CHARS))

    # The classifier is much more efficient if it runs across the entire
    # batch at once, so we want to squash (i.e. uncleave).
    #
    # Hidden nodes is set to 32 because it seems to work well.
    return (lstm
      .squash_sequence()
      .fully_connected(32, activation_fn=tf.nn.relu)
      .dropout(0.7)
      .softmax_classifier(SEXES, labels, per_example_weights=per_example_weights))


def main(_=None):
  print('Starting Baby Names')
  input_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, TIMESTEPS])
  output_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, SEXES])
  inp = data_utils.reshape_data(input_placeholder)

  # Create a label for each timestep.
  lables_1 = tf.reshape(tf.tile(output_placeholder, [1, TIMESTEPS]), [BATCH_SIZE, TIMESTEPS, SEXES])
  labels = data_utils.reshape_data(lables_1, per_example_length=2)
  length_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, 1])

  # We need a dense multiplier for the per example weights.  The only place
  # that has a non-zero loss is the first EOS after the last character of the
  # name; the characters in the name and the trailing EOS characters are given a
  # 0 loss by assigning the weight to 0.0 and in the end only one character in
  # each batch has a weight of 1.0.
  # sparse_to_dense does a lookup using the indices from the first Tensor.
  # Because we are filling in a 2D array, the indices need to be 2 dimensional.
  # Since we want to assign 1 value for each row, the first dimension can just
  # be a sequence.
  t = tf.concat(1, [tf.constant(numpy.arange(BATCH_SIZE).reshape((BATCH_SIZE, 1)),dtype=tf.int32), length_placeholder])

  # Squeeze removes dimensions that are equal to 1.  per_example_weights must
  # end up as 1 dimensional.
  per_example_weights = data_utils.reshape_data(tf.sparse_to_dense(
      t, [BATCH_SIZE, TIMESTEPS], 1.0, default_value=0.0)).squeeze()

  # We need 2 copies of the graph that share variables.  The first copy runs
  # training and will do dropout if specified and the second will not include
  # dropout.  Dropout is controlled by the phase argument, which sets the mode
  # consistently throughout a graph.
  with tf.variable_scope('baby_names'):
    result = create_model(inp, labels, TIMESTEPS, per_example_weights)

  # Call variable scope by name so we also create a name scope.  This ensures
  # that we share variables and our names are properly organized.
  with tf.variable_scope('baby_names', reuse=True):
    # Some ops have different behaviors in test vs train and these take a phase
    # argument.
    test_result = create_model(inp, labels, TIMESTEPS, per_example_weights, phase=pt.Phase.test)

  # For tracking accuracy in evaluation, we need to add an evaluation node.
  # We only run this when testing, so we need to specify that in the phase.
  # Some ops have different behaviors in test vs train and these take a phase
  # argument.
  accuracy = test_result.softmax.evaluate_classifier(
      labels,
      phase=pt.Phase.test,
      per_example_weights=per_example_weights)

  # We can also compute a batch accuracy to monitor progress.
  batch_accuracy = result.softmax.evaluate_classifier(
      labels,
      phase=pt.Phase.train,
      per_example_weights=per_example_weights)

  # Grab the inputs, outputs and lengths as numpy arrays.
  # Lengths could have been calculated from names, but it was easier to
  # calculate inside the utility function.
  names, sex, lengths = data_utils.baby_names(TIMESTEPS)

  epoch_size = len(names) // BATCH_SIZE
  # Create the gradient optimizer and apply it to the graph.
  # pt.apply_optimizer adds regularization losses and sets up a step counter
  # (pt.global_step()) for you.
  # This sequence model does very well with initially high rates.
  optimizer = tf.train.AdagradOptimizer(
      tf.train.exponential_decay(1.0,
                                 pt.global_step(),
                                 epoch_size,
                                 0.95,
                                 staircase=True))
  train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

  # We can set a save_path in the runner to automatically checkpoint every so
  # often.  Otherwise at the end of the session, the model will be lost.
  runner = pt.train.Runner(save_path=FLAGS.save_path)
  with tf.Session():
    for epoch in xrange(100):
      # Shuffle the training data.
      names, sex, lengths = data_utils.permute_data((names, sex, lengths))

      runner.train_model(
          train_op,
          [result.loss, batch_accuracy],
          epoch_size,
          feed_vars=(input_placeholder, output_placeholder, length_placeholder),
          feed_data=pt.train.feed_numpy(BATCH_SIZE, names, sex, lengths),
          print_every=100)
      classification_accuracy = runner.evaluate_model(
          accuracy,
          epoch_size,
          print_every=0,
          feed_vars=(input_placeholder, output_placeholder, length_placeholder),
          feed_data=pt.train.feed_numpy(BATCH_SIZE, names, sex, lengths))

      print('Accuracy after epoch %d: %g%%' % (
          epoch + 1, classification_accuracy * 100))


if __name__ == '__main__':
  tf.app.run()
