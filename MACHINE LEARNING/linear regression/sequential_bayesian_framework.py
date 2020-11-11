#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:21:27 2020

@author: shengkailin
"""


# libraries
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.stats import kde
#%%
def prior_w(w1,w2,alpha,beta):
	return 1/(np.sqrt(2*np.pi))*exp(-1/2*(alpha*w1**2+beta*w2**2))

w1=np.linspace(-3.0,3.0,100)
w2=np.linspace(-3.0,3.0,100)
#meshgrid function is used to generate variable matrix for the function
#to see the intesection element of ith row and jth column use w1[i,j]  
w1, w2 =np.meshgrid(w1, w2)
alpha=1
beta=1
z=prior_w(w1,w2,alpha,beta)
#commen cmap configuration:cividis(uniform),bwr(diverging),jet(miscellaneous)
plt.pcolormesh(w1, w2, z, cmap="jet",shading="gouraud")
plt.colorbar()
plt.title('w prior')
plt.show()
#%% solving 2d gaussian distribution intergrals 
from sympy import *
from IPython.display import display

init_printing(use_unicode=False,wrap_line=False)
x=Symbol('x')
y=Symbol('y')
display(integrate())




