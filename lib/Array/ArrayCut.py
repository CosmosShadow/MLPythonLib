# coding: utf-8
from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import sys
import math

# cutWithLength: 用固定长度来切割
# cutWithMaxValue: 用最大或者最小切割，而且间隔要相近

# data: 数据
# sectionLength: 切割长度
def cutWithLength(data, sectionLength):
	arrOfSections = []
	for i in range(data.shape[0] - sectionLength):
		section = data[i: i+sectionLength]
		arrOfSections.append(section)
	return arrOfSections;


# 用最大或者最小切割，而且间隔要相近
# data: 数据
# sectionLength: 切割的大概间隔
# isMax: True, 用最大值切割，False, 用最小值切割
def cutWithMaxValue(data, sectionLength, isMax):
	localMaxValueIndex = []
	for i in xrange(1, len(data)-1):
		if isMax:
			if data[i] > data[i-1] and data[i] > data[i+1]:
				localMaxValueIndex.append(i)
		else:
			if data[i] < data[i-1] and data[i] < data[i+1]:
				localMaxValueIndex.append(i)
	print localMaxValueIndex

	# smooth split index
	smoothIndex = [localMaxValueIndex[0]]
	for i in xrange(1, len(localMaxValueIndex)):
		if localMaxValueIndex[i] - smoothIndex[-1] < sectionLength/4:
			smoothIndex[-1] = int((localMaxValueIndex[i] + smoothIndex[-1]) / 2)
		else:
			smoothIndex.append(localMaxValueIndex[i])

	sectionIndexArray = []
	sectionIndexToShow = []
	for i in xrange(1, len(smoothIndex)):
		gap = smoothIndex[i] - smoothIndex[i-1]
		if  sectionLength * 2 / 3 < gap and gap < sectionLength * 4 / 3:
			sectionIndexArray.append([smoothIndex[i-1], smoothIndex[i]])
			sectionIndexToShow.append(smoothIndex[i-1])
			sectionIndexToShow.append(smoothIndex[i])

	return (sectionIndexToShow, sectionIndexArray)