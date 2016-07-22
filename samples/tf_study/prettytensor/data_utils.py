# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
import os.path
import sys

import numpy as np
import tensorflow as tf
import prettytensor as pt


UNK = 0
EOS = 1


def convert_to_int(char):
  i = ord(char)
  if i >= 128:
    return UNK
  return i


def shakespeare(chunk_size):
  """Downloads Shakespeare, converts it into ASCII codes and chunks it.

  Args:
    chunk_size: The dataset is broken down so that it is shaped into batches x
      chunk_size.
  Returns:
    A numpy array of ASCII codes shaped into batches x chunk_size.
  """
  file_name = maybe_download('http://cs.stanford.edu/people/karpathy/char-rnn/',
                             'shakespear.txt')
  with open(file_name) as f:
    shakespeare_full = f.read()

  # Truncate the data.
  length = (len(shakespeare_full) // chunk_size) * chunk_size
  if length < len(shakespeare_full):
    shakespeare_full = shakespeare_full[:length]
  arr = np.array([convert_to_int(c) for c in shakespeare_full])[
      0:len(shakespeare_full) / chunk_size * chunk_size]
  return arr.reshape((len(arr) / chunk_size, chunk_size))


def baby_names(max_length=15):
  names = []
  lengths = []
  targets = []
  with open(os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'baby_names.csv'), 'rb') as f:
    first = True
    for l in csv.reader(f, delimiter=','):
      if first:
        first = False
        continue
      assert len(l) == 4, l
      name = l[0]
      if max_length < len(name):
        raise ValueError('Max length is too small: %d > %d' % (max_length, len(name)))
      chars = [convert_to_int(c) for c in name]
      names.append(chars + ([EOS] * (max_length - len(chars))))
      lengths.append([len(name)])
      values = [float(l[2]), float(l[3])]
      if abs(sum(values) - 1) > 0.001:
        raise ValueError('Each row must sum to 1: %s' % l)
      targets.append(values)
  return np.array(names), np.array(targets), np.array(lengths)


def reshape_data(tensor, per_example_length=1):
  """Reshapes input so that it is appropriate for sequence_lstm..

  The expected format for sequence lstms is
  [timesteps * batch, per_example_length] and the data produced by the utilities
  is [batch, timestep, *optional* expected_length].  The result can be cleaved
  so that there is a Tensor per timestep.

  Args:
    tensor: The tensor to reshape.
    per_example_length: The number of examples at each timestep.
  Returns:
    A Pretty Tensor that is compatible with cleave and then sequence_lstm.

  """
  # We can put the data into a format that can be easily cleaved by
  # transposing it (so that it varies fastest in batch) and then making each
  # component have a single value.
  # This will make it compatible with the Pretty Tensor function
  # cleave_sequence.
  dims = [1, 0]
  for i in xrange(2, tensor.get_shape().ndims):
    dims.append(i)
  return pt.wrap(tf.transpose(tensor, dims)).reshape([-1, per_example_length])
