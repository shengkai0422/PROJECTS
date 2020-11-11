#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:28:17 2020

@author: shengkailin
"""


#---------------------------------libraries-----------------------------------#
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

#------program
X=np.random.random(50)
lam=0.73
x=X
Y=np.sin(2*np.pi*X)+np.random.normal(0,0.3)
plt.figure()
plt.plot(X,Y,"bo")
X=np.expand_dims(X,axis=0)
X=np.repeat(X,25,axis=0)
X=np.transpose(X,(1,0))
Y=np.expand_dims(Y,axis=0)
Y=np.transpose(Y,(1,0))
mu=np.linspace(0,1,25)
mu=np.expand_dims(mu,axis=0)
mu=np.repeat(mu,50,axis=0)
Phi=np.exp(-np.power((X-mu),2)*2.5)
w=np.matmul(inv(np.matmul(np.transpose(Phi,(1,0)),Phi)+lam*np.eye(25)),np.transpose(Phi,(1,0)))
w=np.matmul(w,Y)

t=np.matmul(Phi,w)
plt.plot(x,np.squeeze(t),"ro")