# coding: utf-8
from __future__ import division
import numpy as np

# 平滑
# data: 数据
# filterData: 平滑参数，eg. [0.1, 0.2, 04, 0.2, 0.1]

def smoothWithFilter(data, filterData):
	sectionCount = len(filterData)
	count = data.shape[0] - sectionCount +1
	arrResult = np.zeros(count)
	filterTotalValue = np.sum(filterData)
	for i in range(count):
		dataSection = data[i: i+sectionCount]
		smoothValue = np.sum(dataSection * np.array(filterData)) / filterTotalValue
		arrResult[i] = smoothValue
	return arrResult

# 顺着0轴进行平滑
def smoothMatrixAxis0WithFilter(data, filterData):
	sectionCount = len(filterData)
	arrResults = np.zeros((data.shape[0] - sectionCount + 1, data.shape[1]))
	for i in range(data.shape[1]):
		arrResults[:, i] = smoothWithFilter(data[:, i], filterData)
	return arrResults