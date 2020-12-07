#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:40:46 2020

@author: shengkailin
"""

#--------------------------------libraries-------------------------------------
import numpy as np

#----------------------------------classes-------------------------------------
class Logistic_regression():
	
	
	def __init__(self,lr=0.001,n_iters=1000):
		self.lr=lr
		self.n_iters=n_iters
		self.weights=None
		self.bias=None
	
	def fit(self,X,y):
		#init arameters
		n_samples,n_features=X.shape 
		self.weights
		pass
	
	def predict(self,X):
		pass
	
	