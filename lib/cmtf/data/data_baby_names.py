# coding: utf-8
import os
from File.FilePath import *
import tensorflow as tf
import numpy as np
import csv

UNK = 0
EOS = 1

def convert_to_int(char):
  i = ord(char)
  if i >= 128:
    return UNK
  return i

def baby_names(max_length=15):
  data_path = os.environ['HOME'] + '/data/tf/baby_names/baby_names.csv'
  if not os.path.exists(data_path):
  	print 'Error: ' + data_path + ' not exits'
  	return None

  names = []
  lengths = []
  targets = []
  with open(data_path, 'rb') as f:
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