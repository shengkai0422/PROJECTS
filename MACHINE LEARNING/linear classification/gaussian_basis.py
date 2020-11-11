#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:20:52 2020

@author: shengkailin
"""

def gaussian_basis_map(sample,miu,sigma):
	return np.exp(-(np.linalg.norm(sample-miu,axis=1))**2/(2*sigma**2))

import numpy as np
import matplotlib.pyplot as plt
#%%
#generation of 2d gaussian distribution samples
mean=np.array([0,0])
cov=np.array([[0.1,0],[0,0.1]])
sample_red=np.random.multivariate_normal(mean, cov, 100)
plt.scatter(sample_red[:,0],sample_red[:,1],c="red")
mean=np.array([1,1])
cov=np.array([[0.03,0],[0,0.03]])
sample_blue=np.random.multivariate_normal(mean, cov, 100)
plt.scatter(sample_blue[:,0],sample_blue[:,1],c="blue")
mean2=np.array([-1,-1])
cov=np.array([[0.03,0],[0,0.03]])
sample_blue_2=np.random.multivariate_normal(mean2, cov, 100)
plt.scatter(sample_blue_2[:,0],sample_blue_2[:,1],c="blue")
plt.show()

#%%


new_feature_1=gaussian_basis_map(sample_red,np.array([0,0]),0.5)
new_feature_2=gaussian_basis_map(sample_red,np.array([-1,-1]),0.5)
plt.figure()
plt.scatter(new_feature_2,new_feature_1,c="red")

new_feature_1=gaussian_basis_map(sample_blue,np.array([0,0]),0.5)
new_feature_2=gaussian_basis_map(sample_blue,np.array([-1,-1]),0.5)
plt.scatter(new_feature_2,new_feature_1,c="blue")


new_feature_1=gaussian_basis_map(sample_blue_2,np.array([0,0]),0.5)
new_feature_2=gaussian_basis_map(sample_blue_2,np.array([-1,-1]),0.5)
plt.scatter(new_feature_2,new_feature_1,c="blue")

plt.show()
#%%