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

for iteration in range(1, num_iterations+1):
    start_time = time.time()
    data = np.arange(0, 10000 * iteration, 1)
    
    sum = 0 
    for i in data: 
        sum += i 
    end_time = time.time()
    
    print('{} - {}'.format(iteration, end_time - start_time))
    time_results_loop.append(end_time - start_time)
    
    
## optimalization
for iteration in range(1, num_iterations+1):
    start_time = time.time()
    
    data = np.arange(0, 10000 * iteration, 1)
    
    ## difference
    sum = np.sum(data) 
  
    end_time = time.time()
    
    print('{} - {}'.format(iteration, end_time - start_time))
    time_results_loop.append(end_time - start_time)
