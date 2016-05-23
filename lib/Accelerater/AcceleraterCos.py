# coding: utf-8
import numpy as np
import math

def cosOfTwoAcceleraters(one, two):
	cross = one[0]*two[0] + one[1]*two[1] + one[2]*two[2]
	valueX = math.sqrt(one[0]*one[0] + one[1]*one[1] + one[2]*one[2])
	valueY = math.sqrt(two[0]*two[0] + two[1]*two[1] + two[2]*two[2])
	base = (valueX * valueY)
	if base <= 0:
		return 0
	else:
		return cross / base

def cosOfAcceleratersArray(data):
	arrResult = np.zeros(data.shape[0]-1)
	for i in range(data.shape[0]-1):
		cosValue = cosOfTwoAcceleraters(data[i, :], data[i+1, :])
		arrResult[i] = cosValue
	return arrResult

def unitOfAcceleratersArray(data):
	return np.sqrt(np.sum(data * data, axis=1))