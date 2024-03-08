#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:07:49 2024

@author: darekdajcz
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
 
iris = pd.read_csv(r"/Users/darekdajcz/Desktop/Uczenie_maszynowe_Python/Kurs ML/iris.csv",
                   header = None, 
                   names = ['petal length', 'petal width', 
                            'sepal length', 'sepal width', 'species'])



iris.head()