# coding: utf-8
import numpy as np
import pandas as pd

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

print df['A']
print df['A'].values