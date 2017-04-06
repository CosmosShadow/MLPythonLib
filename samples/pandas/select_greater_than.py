# coding: utf-8
import pandas as pd
import numpy as np

a = pd.DataFrame(np.array([1, 2, 3]), index=['A', 'B', 'C'])
print a[a[0] > 1]