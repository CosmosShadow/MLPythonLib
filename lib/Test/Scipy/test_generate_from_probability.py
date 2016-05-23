# coding: utf-8
from scipy import stats
import numpy as np

xk = np.arange(7)
print xk
pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
custm = stats.rv_discrete(name='custm', values=(xk, pk))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
plt.show()

R = custm.rvs(size=100)
print R

for x in xrange(1,100):
	print custm.rvs()