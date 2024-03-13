#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:23:26 2024

@author: darekdajcz
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron: 
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):
        self.epochs = epochs
        self.eta = eta
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    def predict(self, x):
        ## Mnozenie macierzy np.dot
        total_stimulation_Z = np.dot(x, self.w)
        y_pred = 1 if total_stimulation_Z > 0 else -1
        return y_pred
    
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)
        
        self.w = np.random.rand(X_1.shape[1])
        
        for e in range(self.epochs):
            
            errors_counter = 0
            
            for x, y_target in zip(X_1, y):
                
                y_pred = self.predict(x)
                delta_w = self.eta * (y_target - y_pred) * x
                self.w += delta_w
        
                errors_counter += 1 if y_target != y_pred else 0
                
            self.list_of_errors.append(errors_counter)
            
            if(self.is_verbose):
                print("Epoch: {}, weights: {}, errors_counter => {}".format(
                    e, self.w, errors_counter))

X = np.array([
    [2,4,20],
    [4,3,-10],
    [5,6,13],
    [5,4,8],
    [3,4,5],
    [-4, 9, 18],
    [4, 0, -3],
    [18, 10, -4],
    [15, 8, 12],
    [0, 0, 13],
    [10, -7, -3],
    [13, -8, 11],
    [10, -9, 13],
    [1, 19, -5],
    [-9, 17, 10],
    [-10, 1, 15],
    [11, 18, 1],
    [14, 6, 16],
    [16, -1, 17],
    [17, 5, 4],
    [19, 19, 4],
    [19, 8, 1],
    [12, 9, 14],
    [-8, -6, 8],
    [-4, 10, -2]
    ])

y = np.array([1,-1,-1,1,-1,-1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1])

perceptron = Perceptron(eta=0.1, epochs=100, is_verbose=True)
perceptron.fit(X, y)
print(perceptron.w)


print(perceptron.predict(np.array([[1, 2, 3, 1]])))
print(perceptron.predict(np.array([[2, 2, 8, 1]])))
print(perceptron.predict(np.array([[3, 3, 3, 1]])))

%matplotlib inline

plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)

