#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:34:56 2023

@author: alpha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"/Users/alpha/Desktop/DATA SCIENCE/jan25th/SIMPLE LINEAR REGRESSION/slr/Salary_Data.csv")


x = data.iloc[:,:-1]
y = data.iloc[:,1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 1/3,random_state = 0,)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)

ypredct = regressor.predict(xtest)

# training visualiation
plt.scatter(xtrain,ytrain,color = 'red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title("salary Vs Experience (training set)")
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()


# testing visuailizattion
plt.scatter(xtest, ytest,color = 'red')
plt.plot(xtrain,regressor.predict(xtrain),color = 'blue')
plt.title("salary vs experience(testing set)")
plt.xlabel("Yeras of experience")
plt.ylabel*("salary")
plt.show()