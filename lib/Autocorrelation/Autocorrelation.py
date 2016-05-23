# coding: utf-8

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.mlab import find
from scipy.signal import fftconvolve
from parabolic import parabolic

# frequencey: 频率
# strength: 强度
# corr: 相关性
# d: 相关性变化情况
def Autocorrelation(data):
	corr = fftconvolve(data, data[::-1], mode='full')	# 傅立叶卷积
	corr = corr[len(corr)/2:]	# 对称取一半
	d = np.diff(corr)			# 相关性变化情况

	if len(find(d > 0)) <= 0:
		return (-1, -1, None, None)

	start = find(d > 0)[0]
	peak = np.argmax(corr[start:]) + start		# 相关性最大的位移
	frequencey, py = parabolic(corr, peak)				# 加权平均位移
	strength = np.max(d)					# 变化最大值: 强度

	return (int(frequencey), strength, corr, d)
	# return (int(peak), strength, corr, d)

# 最大相关轴
def AutocorrelationBestAxis(data):
	bestIndex = -1
	bestFrequency = -1
	bestStrength = -1
	bestCorr = None
	bestD = None
	for i in range(data.shape[1]):
		(frequencey, strength, corr, d) = Autocorrelation(data[:, i])
		if strength > bestStrength:
			bestIndex = i
			bestFrequency = frequencey
			bestStrength = strength
			bestCorr = corr
			bestD = d
	return (bestIndex, bestFrequency, bestStrength, bestCorr, bestD)



