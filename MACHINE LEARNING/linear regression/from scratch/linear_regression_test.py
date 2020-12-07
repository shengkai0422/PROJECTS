#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:18:40 2020

@author: shengkailin
"""


#-----------------------------libriries---------------------------------------
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from linear_regression import linear_regression
#--------------------------------MSE function---------------------------------
def MSE(y_true,y_predicted):
	
	return np.mean((y_true-y_predicted)**2)
#-------------------------------data------------------------------------------
X,y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)


regressor=linear_regression(lr=0.001)
regressor.fit(X_train,y_train)
predicted= regressor.predict(X_test)
pre_line=regressor.predict(X)
mse_value=MSE(y_test,predicted)
print(mse_value)
fig=plt.figure(figsize=(8,6))
plt.scatter(X[:,0],y,color='b',marker='o',s=30)
plt.plot(X,pre_line,color='black',linewidth=2,label='prediction')
plt.show()

regressor=linear_regression(lr=0.01)
regressor.fit(X_train,y_train)
predicted= regressor.predict(X_test)
pre_line=regressor.predict(X)
mse_value=MSE(y_test,predicted)
print(mse_value)
fig=plt.figure(figsize=(8,6))
plt.scatter(X[:,0],y,color='b',marker='o',s=30)
plt.plot(X,pre_line,color='black',linewidth=2,label='prediction')
plt.show()

