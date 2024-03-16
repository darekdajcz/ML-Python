#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:23:26 2024

@author: darekdajcz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class Perceptron: 
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):
        self.epochs = epochs
        self.eta = eta
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    def predict(self, x):
        ones = np.ones((x.shape[0], 1))
        x_1 = np.append(x.copy(), ones, axis=1)
        return self.__predict(x_1)
        
        
    def __predict(self, x):
        ## Mnozenie macierzy np.dot
        total_stimulation_Z = np.dot(x, self.w)
        y_pred_v = np.where(total_stimulation_Z > 0, 1, -1)
        return y_pred_v
    
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)
        
        self.w = np.random.rand(X_1.shape[1])
        
        for e in range(self.epochs):
            
            errors_counter = 0
    
                
            y_pred_v = self.__predict(X_1)
            delta_w_v = self.eta * np.dot((y - y_pred_v), X_1)
            self.w += delta_w_v
        
            errors_counter = np.count_nonzero(y- y_pred_v)
                
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
perceptron.fit(X, y) #training
print(perceptron.w)


print(perceptron.predict(np.array([[1, 2, 3]])))
print(perceptron.predict(np.array([[2, 2, 8]])))
print(perceptron.predict(np.array([[3, 3, 3]])))

%matplotlib inline

plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)



# predict for Iris Flowers
df = pd.read_csv(r"/Users/darekdajcz/Desktop/Uczenie_maszynowe_Python/Kurs_ML/iris_diagrams/csv_data/iris.csv",
                   header = None)
df

df = df.iloc[:100, :].copy()
df[4] = df[4].apply(lambda x: 1 if x == 'Iris-setosa' else -1)
df


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

p = Perceptron(eta = 0.05, epochs=50)
p.fit(X_train, y_train)

y_pred = p.predict(X_test)
print(list(zip(y_pred, y_test)))
print(np.count_nonzero(y_pred - y_test))


