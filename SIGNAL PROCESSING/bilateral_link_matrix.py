#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:20:42 2020

@author: shengkailin
"""


#---------------------------------libraries-----------------------------------
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#----------------------------------classes------------------------------------
#think about defining a pipeline(map-reduce)，decorator

class Bilateral_linkage_matrix():
	
	def __init__(self,dim=5):
		
		self.dim=dim
		#W=np.zeros((self.dim,self.dim))
	
	@staticmethod
	def plot_xlogx():
		#to see if function z=-xlogx-ylogy is concave(凹函数)
		#concave function has a unique maximum/minimum
		fig=plt.figure()
		ax=Axes3D(fig)
		x=np.linspace(0,1,100)
		y=np.linspace(0,1,100)
		#build up a grid 
		X,Y=np.meshgrid(x,y)
		Z=-X*np.log(X)-Y*np.log(Y)
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title(r"z=-xlog(x)-ylog(y)")
		ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
		plt.show()

	@staticmethod
	def entropy(self,W):
		W=np.reshape(W,self.dim**2)
		return sum(-W*np.log(W))
		
	def prior_matrix(self,W,conditions):
		pass
		#using lagrange multiplier method 
		#fun=entropy(self,W)
		
		
		
	
	
#monScript.py
if __name__ == '__main__':
	Bilateral_linkage_matrix.plot_xlogx()