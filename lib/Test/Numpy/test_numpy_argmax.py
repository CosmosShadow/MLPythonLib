# coding: utf-8

import numpy as np

a = np.array([[1, 2, -1, 4], [8, 7, 6, 5]])

sorted = a.argsort()
print sorted

print sorted[:, -1]