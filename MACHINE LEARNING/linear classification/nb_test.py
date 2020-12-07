#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:45:12 2020

@author: shengkailin
"""


#------------------------------libraries--------------------------------------#
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
#import our NavieBayes model
from nb import NaiveBayes
#------------------------------------------------------------------------------
#---------------------------------functions-----------------------------------#
def accuracy(y_true,y_pred):
	accuracy=np.sum(y_true==y_pred)/len(y_true)
	return accuracy 

#-----------------------------------------------------------------------------
#-----------------------------------test program----------------------------------#
X,y=datasets.make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=123)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=123)

nb=NaiveBayes()
nb.fit(X_train,y_train)
prediction=nb.predict(X_test)
print("Naive Bayes classification accuracy",accuracy(y_test,prediction))
