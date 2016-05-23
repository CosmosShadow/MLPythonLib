# coding: utf-8

import numpy as np

# frequencyCount: 频数
# 输入: [1, 2, 1, 3]
# 输出: [(1, 2), (2, 1), (3, 1)]
def frequencyCount(data):
	y = np.bincount(data)
	ii = np.nonzero(y)[0]
	fc = zip(ii,y[ii])
	return fc

# frequencyCountMoreThan: 频数大于一定值的值
def  frequencyCountMoreThan(data, minCount):
	fc = frequencyCount(data)
	fcmt = []
	for (value, count) in fc:
		if count > minCount:
			fcmt.append(value)
	return fcmt

# 最大计数
def frequencyCountMaxValue(data):
	fc = frequencyCount(data)
	maxCountValue = 0
	maxCount = 0
	for (value, count) in fc:
		if count > maxCount:
			maxCount = count
			maxCountValue = value
	return (maxCountValue, maxCount)
