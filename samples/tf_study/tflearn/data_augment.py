# coding: utf-8
import tensorflow as tf
from tflearn.data_augmentation import *
import numpy as np
from scipy import misc
import random

# 读图片
im = misc.imread('1.png')
im = im.reshape((1, 28, 28, 1))
print im.shape

# # 数据增强器
# imageAug = ImageAugmentation()
# imageAug.add_random_crop((28, 28), 2)

# # 数据增强
# im = imageAug.apply(im)

s1 = random.randint(0, 2)
s2 = random.randint(0, 2)
s3 = random.randint(0, 2)
s4 = random.randint(0, 2)
l = 26
im[:,s1:s1+l,s2:s2+l,:] = im[:,s3:s3+l,s4:s4+l,:]

# 存图片
im = im[0].reshape((28, 28))
misc.imsave('2.png', im)

# for _ in xrange(1,10):
	# print 