#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:40:47 2024

@author: darekdajcz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
%matplotlib inline
 
class Perceptron:
    
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):
        
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    
    def predict(self, x):
        
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)
        #activation = self.get_activation(x_1)
        #y_pred = np.where(activation >0, 1, -1)
        #return y_pred
        return np.where(self.get_activation(x_1) > 0, 1, -1)
        
    
    def get_activation(self, x):
        
        activation = np.dot(x, self.w)
        return activation
     
    
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)
 
        self.w = np.random.rand(X_1.shape[1])
        
        for e in range(self.epochs):
 
            error = 0
            
            activation = self.get_activation(X_1)
            delta_w = self.eta * np.dot((y - activation), X_1)
            self.w += delta_w
                
            error = np.square(y - activation).sum()/2.0 #minimalizacja błedu kwadratowego
                
            self.list_of_errors.append(error)
            
            if(self.is_verbose):
                print("Epoch: {}, weights: {}, error {}".format(
                        e, self.w, error))
                
                
                
X = np.array([
    [3., 5.,  23.], 
    [4., 3., -12.], 
    [5., 6.,  11.], 
    [5., 4.,   9.], 
    [3., 4.,   7.],  
])
 
y = np.array([1, -1, -1, 1, -1])
 
perceptron = Perceptron(eta=0.0001, epochs=100, is_verbose=True)            
perceptron.fit(X, y)
plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)
perceptron.predict(np.array([ [3., 5.,  -23.]]))
 
 
df = pd.read_csv(r"/Users/darekdajcz/Desktop/Uczenie_maszynowe_Python/Kurs_ML/iris_diagrams/csv_data/iris.csv",
                   header = None)
df = df.iloc[:100, :].copy()
df[4] = df[4].apply(lambda x: 1 if x == 'Iris-setosa' else -1)
df
 
X = df.iloc[0:100, :-1].values
y = df[4].values
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
 
iris_perceptron = Perceptron(eta = 0.0001, epochs=100) 
iris_perceptron.fit(X_train, y_train)   
             
y_pred = iris_perceptron.predict(X_test)
 
plt.scatter(range(p.epochs), iris_perceptron.list_of_errors)


iris_perceptron.predict(np.array([ [5.6, 2.6,  4.3, 1.2]]))
