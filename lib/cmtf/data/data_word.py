# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from File.FilePath import *

def maybe_download():
  download_url = 'http://mattmahoney.net/dc/'
  file_dir = os.environ['HOME'] + '/data/tf/word/'
  filename = os.environ['HOME'] + '/data/tf/word/text8.zip'
  NoExistsCreateDir(file_dir)
  if not os.path.exists(filename):
    urllib.request.urlretrieve(download_url, filename)
  return filename

def words_src():
  filename = maybe_download()
  with zipfile.ZipFile(filename) as f:
    wds = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return wds

def words_encode(vocabulary_size):
  words = words_src()
  print('Data size', len(words))
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

if __name__ == '__main__':
  data, count, dictionary, reverse_dictionary = words_encode(50000)
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])



