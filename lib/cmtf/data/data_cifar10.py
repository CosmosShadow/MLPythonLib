# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import re
import sys
import tarfile
from six.moves import urllib
from File.FilePath import *
import tensorflow as tf
import cmtf.data.data_cifar10_google as data_cifar10_google

# 超参
IMAGE_SIZE = data_cifar10_google.IMAGE_SIZE
NUM_CLASSES = data_cifar10_google.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_cifar10_google.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_cifar10_google.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

def maybe_download_and_extract(DATA_URL, data_dir):
  """Download and extract the tarball from Alex's website."""
  dest_directory = data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def path_of_data():
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
  data_path = os.environ['HOME'] + '/data/tf/cifar10/'
  NoExistsCreateDir(data_path)
  maybe_download_and_extract(DATA_URL, data_path)
  return data_path

def inputs(eval_data, batch_size):
  data_path = path_of_data()
  return data_cifar10_google.inputs(eval_data, data_path, batch_size)

def distorted_inputs(batch_size):
  data_path = path_of_data()
  return data_cifar10_google.distorted_inputs(data_path, batch_size)