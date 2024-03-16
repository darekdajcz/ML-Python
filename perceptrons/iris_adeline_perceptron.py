#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:40:47 2024

@author: darekdajcz
"""

import numpy as np
 
class Perceptron:
    
    def __init__(self, eta=0.10, epochs=50):
        
        self.eta = eta
        self.epochs = epochs
        
    
    def predict(self, x):
        
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)
        return np.where(self.......A......(x_1) > 0, 1, -1)
        
    
    def get_activation(self, x):
        
        activation = np.dot(......B......, self.w)
        return activation
     
    
    def fit(self, X, y):
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)
 
        self.w = np.random.rand(X_1.shape[1])
        
        for e in range(self.epochs):
            
            activation = self.get_activation(X_1)
            delta_w = self.eta * np.dot((y - ......C......), X_1)
            self.w += delta_w