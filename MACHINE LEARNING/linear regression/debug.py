#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:52:06 2020

@author: shengkailin
"""


#import pdb
#pdb.set_trace()
#Gaussian basis functions are not built into Scikit-Learn,
#but we can write a custom transformer that will create them, 
#as shown here and illustrated in the following figure
#(Scikit-Learn transformers are implemented as Python classes; 
#reading Scikit-Learn's source is a good way to see how they can be created

from scipy.stats import reciprocal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt


# a new class gaussianfeatures （duck typing）
#use BaseEstimator as base class :no *arg and **kwarg in the __init__ 
#use TransformerMixin as bese classs :automatically done fit_transform <=>fit.().transform()
class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
        
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        #返回self，确保在转换器中能够进行链式调用（例如调用transformer.fit(X).transform(X)）
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,

                                 self.width_, axis=1)
   
#%%
xfit = np.linspace(0, 10, 1000)
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
gauss_model = make_pipeline(GaussianFeatures(20),
                            LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10);