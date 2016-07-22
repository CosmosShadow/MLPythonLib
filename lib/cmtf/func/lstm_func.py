# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np

# 把数据转成lstm识别的格式: batch x step x data -> step*batch x data
def reshape_data_to_lstm_format(tensor, per_example_length=1):
  dims = [1, 0]
  for i in xrange(2, tensor.get_shape().ndims):
    dims.append(i)
  return pt.wrap(tf.transpose(tensor, dims)).reshape([-1, per_example_length])