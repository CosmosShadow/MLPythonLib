# coding: utf-8
import numpy as np

# 数据的频率、振幅
def freqOfArray(data):
	# fft
	fftValue = abs(np.fft.fft(data))
	# 去掉了基底与1的振幅
	fftValue[0] = 0
	fftValue[1] = 0
	# 输出最大量、百分比
	halfData = fftValue[0:fftValue.shape[0]/2]
	maxIndex = halfData.argmax()
	freq = fftValue.shape[0] / maxIndex
	maxPercent = halfData[maxIndex] / sum(halfData)
	# 返回: 振幅最大的频率长度、相应振幅所占百分比(去掉了基底与1的振幅)
	return (freq, maxPercent)
