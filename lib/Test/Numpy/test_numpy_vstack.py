# coding: utf-8

import numpy as np
data = np.ones((1, 2, 2, 2))

append_data = np.zeros((2, 2, 2, 2))

print data.shape
print append_data.shape

c = np.vstack([data, append_data])

print c.shape

