# coding: utf-8
from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import sys
import math


def rangeWithValues(arr, values):
	startIndex = 0
	endIndex = len(arr)-1

	for i in range(len(arr)):
		if arr[i] in values:
			startIndex = i
			break
	for i in xrange(len(arr)-1, -1, -1):
		if arr[i] in values:
			endIndex = i
			break

	return (startIndex, endIndex)

if __name__ == "__main__":
	arr = [1, 1, 2, 2, 3, 5, 6, 2, 0, 10]
	values = [2, 0]
	print rangeWithValues(arr, values)	#output: (2, 8)