# coding: utf-8

import numpy as np
import math

a = np.array([[1, 2, -1, 4], [8, 7, 6, 5]])
print a
print ''

max_index = np.argmax(a, axis=1)
print max_index
print ''

max_value = np.max(a, axis=1)
print max_value
print ''

x = np.arange(10)
print x
print x[:5]
print ''

print x[1:3]
print np.prod(x[1:3])
print ''

print math.pow(9, 0.5)
print ''

b = np.zeros(2)
max_index = np.argmax(b)
print max_index