#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:29:45 2024

@author: darekdajcz
"""

import time 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

num_iterations = 100
time_results_loop = []
time_results_np = []

## np optimalization with sum
for iteration in range(1, num_iterations+1):
    start_time = time.time()
    data = np.arange(0, 10000 * iteration, 1)
    
    totalSum = 0 
    for i in data: 
        totalSum += i 
    end_time = time.time()
    
    print('{} - {}'.format(iteration, end_time - start_time))
    time_results_loop.append(end_time - start_time)
    
    

for iteration in range(1, num_iterations+1):
    start_time = time.time()
    
    data = np.arange(0, 10000 * iteration, 1)
    
    ## difference
    totalSum = np.sum(data) 
  
    end_time = time.time()
    
    print('{} - {}'.format(iteration, end_time - start_time))
    time_results_np.append(end_time - start_time)
    


fig = plt.figure()
plt.scatter(range(100), time_results_loop, s=10, c='b', marker="s", label='loop')
plt.scatter(range(100), time_results_np, s=10, c='r', marker="o", label='numpy')
plt.legend(loc='lower right');
plt.show()

num_iterations = 100
time_results_loop = []
time_results_np = []

## np optimalization with sum matrix
for iteration in range(1, num_iterations+1):
    start_time = time.time()
    
    data1 = np.ones(shape=(10*iteration, 10*iteration), dtype=float)
    data2 = np.ones(shape=(10*iteration, 10*iteration), dtype=float)
    data3 = np.zeros(shape=(10*iteration, 10*iteration), dtype=float)
    
    for i in range(data1.shape[0]):  # Dla każdego wiersza
        for j in range(data1.shape[1]):  # Dla każdej kolumny
            data3[i, j] = data1[i, j] + data2[i, j]  # Sumuj odpowiadające sobie elementy i zapisz w data3
    

    end_time = time.time()
    
    print('{} - {}'.format(iteration, end_time - start_time))
    time_results_loop.append(end_time - start_time)
    
    

for iteration in range(1, num_iterations+1):
    start_time = time.time()
    
    data1 = np.ones(shape=(10*iteration, 10*iteration), dtype=float)
    data2 = np.ones(shape=(10*iteration, 10*iteration), dtype=float)
    data3 = np.zeros(shape=(10*iteration, 10*iteration), dtype=float)
    
    ## difference
    data3 = data1 + data2

    end_time = time.time()
    
    print('{} - {}'.format(iteration, end_time - start_time))
    time_results_np.append(end_time - start_time)
    

fig = plt.figure()
print(len(time_results_np))
print(len(time_results_loop))
plt.scatter(range(len(time_results_np)), time_results_loop, s=10, c='b', marker="s", label='loop')
plt.scatter(range(len(time_results_np)), time_results_np, s=10, c='r', marker="o", label='numpy')
plt.legend(loc='lower right')
plt.show()


## np optimalization with multiply matrix
num_iterations = 20
time_results_loop = []
 
for iteration in range(1, num_iterations+1):
    
    start_time = time.time()
    
    data1 = np.ones(shape=(10*iteration, 10*iteration), dtype=float)
    data2 = np.ones(shape=(10*iteration, 10*iteration), dtype=float)
    data3 = np.zeros(shape=(10*iteration, 10*iteration), dtype=float)
   
    for i in range(data1.shape[0]): # i - row number
        for j in range(data2.shape[1]): # j - column number
            data3[i,j] = np.sum([data1[i, v] * data2[v, j] for v in range(data1.shape[1])])
    
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_loop.append(end_time - start_time)
 
 
 
num_iterations = 20
time_results_np = []
 
for iteration in range(1, num_iterations+1):
    
    start_time = time.time()
    
    data1 = np.ones(shape=(10*iteration, 10*iteration), dtype=float)
    data2 = np.ones(shape=(10*iteration, 10*iteration), dtype=float)
    data3 = np.zeros(shape=(10*iteration, 10*iteration), dtype=float)
   
    data3 = data1.dot(data2)
    
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_np.append(end_time - start_time)
 
 
 
 
 
fig = plt.figure()
plt.scatter(range(num_iterations), time_results_loop, s=10, c='b', marker="s", label='loop')
plt.scatter(range(num_iterations), time_results_np, s=10, c='r', marker="o", label='numpy')
plt.legend(loc='upper left');
plt.show()