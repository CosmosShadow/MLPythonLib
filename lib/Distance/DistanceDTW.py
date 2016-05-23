# coding: utf-8
import numpy as np
import sys
import math

def ds(x, y):
    return (x - y) ** 2

def DTW(x, y, dist=lambda x, y: norm(x - y, ord=1)):
    x = np.array(x)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    y = np.array(y)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    r, c = len(x), len(y)

    D = np.zeros((r + 1, c + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf

    for i in range(r):
        for j in range(c):
            D[i+1, j+1] = dist(x[i], y[j])

    for i in range(r):
        for j in range(c):
            D[i+1, j+1] += min(D[i, j], D[i, j+1], D[i+1, j])

    D = D[1:, 1:]

    dist = D[-1, -1] / sum(D.shape)

    return dist

# multi Dimension
def dtw_seq(s1, s2, num_s):
    if num_s == 1:
        d = DTW(s1, s2, dist = ds)
    else:
        l = len(s1)
        dist_square_sum = 0
        for i in range(l):
            dist_square_sum += DTW(s1[i], s2[i], dist = ds) ** 2
        d = np.sqrt(dist_square_sum)
    return d

def DTWOfArrayToMatrix(elements, num_s):
    count = len(elements)
    distanceMat = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            if i < j:
                distance = dtw_seq(elements[i], elements[j], num_s)
                distanceMat[i, j] = distance
                distanceMat[j, i] = distance
    return distanceMat

def DTWOfTwoArraysToMatrix(arr1, arr2, num_s):
    count1 = len(arr1)
    count2 = len(arr2)
    distanceMat = np.zeros((count1, count2))
    for i in range(count1):
        for j in range(count2):
                distance = dtw_seq(arr1[i], arr2[j], num_s)
                distanceMat[i, j] = distance
    return distanceMat


if __name__ == "__main__":
    # A = [1,2,2,3,3,4,4]
    A = [1,2,3,4,5,4,3,2,1]
    B = [1,2,3,4,5]
    d = DTW(A, B, dist = ds)
    print d

