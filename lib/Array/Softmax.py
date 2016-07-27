# coding: utf-8

import numpy as np

def softmax(data, dim):
	if dim == 0:
		energe = np.exp(data)
		energe_sum = np.sum(energe, 0)
		possibility = energe / np.tile(energe_sum.reshape(1, -1), (data.shape[0], 1))
		return possibility

	if dim == 1:
		energe = np.exp(data)
		energe_sum = np.sum(energe, 1)
		possibility = energe / np.tile(energe_sum.reshape(-1, 1), (1, data.shape[1]))
		return possibility

	print 'Error: dim wrong'