#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:56:34 2024

@author: darekdajcz
"""

import numpy as np
 
data = np.array([[10, 7, 4], [3, 2, 1]])
data

meanData = np.mean(data)
meanDataColumn = np.mean(data, axis=0)
meanDataVerse = np.mean(data, axis=1)

averageData = np.average(data)

averageDataColumn = np.average(data, axis=0)
averageDataVerse = np.average(data, axis=1)

averageDataWeight1 = np.average(data, axis=1, weights=[0,1,1])
averageDataWeigh2 = np.average(data,  axis=1, weights=[2, 3, 5])

varianceData = np.var(data)
varianceDataColumn = np.var(data, axis=0)
varianceDataVerse = np.var(data, axis=1)

stdData = np.std(data)
stdDataColumn = np.std(data, axis=0)
stdDataVerse = np.std(data, axis=1)

data = np.zeros((2, 1000000))
data[0, :] = 1.0
data[1, :] = 0.1
np.mean(data, dtype=np.float32)
np.mean(data, dtype=np.float64)
data

data = np.zeros((2, 10))
data[0, :] = 1.0
data[1, :] = 0.1
np.mean(data, dtype=np.float32)
np.mean(data, dtype=np.float64)
