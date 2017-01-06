# coding: utf-8
import numpy as np

def images2one(data, padsize=1, padval=1.0):
	n = int(np.ceil(np.sqrt(data.shape[0])))
	h = n * data.shape[1] + (n - 1)
	w = n * data.shape[2] + (n - 1)
	unit_shape = [h, w] + list(data.shape)[3:]
	unit = np.ones(unit_shape, dtype=data.dtype) * padval
	for i in range(data.shape[0]):
		h_i = (i / n) * (data.shape[1] +1)
		w_i = (i % n) * (data.shape[2] + 1)
		unit[h_i: h_i+data.shape[1], w_i: w_i + data.shape[2] ] = data[i]
	return unit