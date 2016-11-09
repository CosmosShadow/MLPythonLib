# coding: utf-8
from scipy import misc

img = misc.imread('1.png', flatten=True)
print img.shape
misc.imsave('2.png', img)