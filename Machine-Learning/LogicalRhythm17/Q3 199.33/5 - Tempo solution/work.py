#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:15:11 2017

@author: Himanshu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing datasets
dataset = pd.read_csv('train.csv')
dataset = dataset.fillna(-1)
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''
#Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regD = DecisionTreeRegressor(random_state = 0)
regD.fit(X_train,y_train)
'''

#Fitting the RandomForestRegressor to the dataset
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
reg.fit(X_train,y_train)


#Predict
y_pred = reg.predict(X_test)

#Accuracy check
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)*200


#Final test
d2 = pd.read_csv('test.csv')
d2 = d2.fillna(-1)
X2 = d2.iloc[:, 3:].values
y_pred2 = reg.predict(X2)

#To save as csv
y_pred2.tofile('foo.csv',sep='\n',format='%10.5f')