#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:03:56 2020

@author: shengkailin
"""

#------------------------------libraries---------------------------------#
import numpy as np
#-------------------------------------------------------------------------
class NaiveBayes():
	
	def fit(self,X,y):
		#samples on row ,colums are feature 
		n_samples,n_features=X.shape
		self._classes=np.unique(y)
		n_classes=len(self._classes)
		
		#init mean var priors
		self._mean=np.zeros((n_classes,n_features),dtype=np.float64) 
		self._var=np.zeros((n_classes,n_features),dtype=np.float64) 
		self._priors=np.zeros(n_classes,dtype=np.float64) 
		
		for cls in self._classes:
			X_cls=X[cls==y]
			#fill this row and all colunms 
			self._mean[cls,:]=X_cls.mean(axis=0)
			self._var[cls,:]=X_cls.var(axis=0)
			#prior of class is equal to the frequency of this class
			self._priors[cls]=X_cls.shape[0] / float(n_samples)

	def predict(self,X):
		y_pred=[self._predict(x) for x in X]
		return y_pred
		
	#build a helper function 
	def _predict(self,x):
		posteriors=[]
		
		for idx,c in enumerate(self._classes):
			prior=np.log(self._priors[idx])
			class_conditional=np.sum(np.log(self._pdf(idx,x)))
			posterior=prior+class_conditional
			posteriors.append(posterior)
		
		return self._classes[np.argmax(posteriors)]
		
	# a helper function for _predict 
	def _pdf(self,class_idx,x):
		mean=self._mean[class_idx]
		var=self._var[class_idx]
		#gaussian density function
		numerator=np.exp(-(x-mean)**2/(2*var))
		denominator=np.sqrt(2*np.pi*var)
		return numerator/denominator
		