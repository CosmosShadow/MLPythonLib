# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:57:42 2015

@author: xuanliang
"""

import numpy as np

def transformY(y):
    numclass = len(np.unique(y))
    l = len(y)
    transy = np.zeros((l, numclass), dtype = int)
    for i in range(l):
        indcol = y[i]
        transy[i, indcol] = 1
    return transy.reshape(len(transy), len(transy[0]), 1)
    
def transformX(x):
    return x.reshape(len(x), len(x[0]), 1)    

def genrModelData(x, y):
    l = len(x)
    transx = transformX(x)
    transy = transformY(y)
    modeldata = list()
    for i in range(l):
        modeldata.append((transx[i], transy[i]))
    return modeldata

def genrTestData(x, y):
    l = len(x)
    transx = transformX(x)
    testdata = list()
    for i in range(l):
        testdata.append((transx[i], y[i]))
    return testdata

if __name__ == '__main__':
    x = np.array([[1,2,3],
                  [3,2,1]])
    y = np.array([0,1])

    print transformY(y)
    print transformX(x)

    print genrModelData(x, y)
    print genrTestData(x, y)