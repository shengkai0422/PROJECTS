#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:38:11 2020

@author: Shengkai Lin
"""
#-------------------------------read me---------------------------------------#
#This program is to visualize the famous bias-variance trade-off effect:
#more speficically supose the a serie of dataset denoted as D_{i}(i=1,2,3,...)
# is generated from a distribution p(t,x).The bias is defined as the average 
#the extent to which the average prediction over all data sets h(x) differs 
#from the desired regression function. The math experssion is then:
#E_{D}{y(x,D_{i}}-h(x). The h(x),is the optimal solution of y(x)
#which minimizes the   the average error:SS(y(x)-t)^2p(x,t)dxdt  
# can be resolved by using variational method.
#E_{D}means the average oprator is on the different dataset D_{i}(i=1,2,3...))

#The h(x)function is chosen to be a sinus function h(x)=sin(2pix), in the real 
#application this value can't be 
#generation of M dataset each consist of N points
#The input values { x_n} are generated uniformly in range (0,1)
#and the corresponding target values{t_n} are obtained first by 
#computing the corresponding values of the function sin(2pix),and then
#adding random noise with a Gaussian distribution having standard
#deviation 0.4



#---------------------------------libraries-----------------------------------#

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from textwrap import wrap



#----------------------------------class---------------------------------------# 
#data class can generate X Y position and display any possible dataset 

class Data():
	

	
	X=None
	Y=None
	set_size=None
	set_volume=None
	mu=None
	sigma=None
	
	def __init__(self,X=None,Y=None,set_size=None,set_volume=None,mu=None,sigma=None):
		self.X=X
		self.Y=Y
		self.set_size=set_size
		self.set_volume=set_volume
		self.mu=mu
		self.sigma=sigma
		print("class data object construced!\n")
	


	
	def generator(self):
		
		#self.X=np.random.random((self.set_size,self.set_volume))
		self.X=np.linspace(0,1,self.set_volume)
		self.X=np.expand_dims(self.X,axis=0)
		self.X=np.repeat(self.X,self.set_size,axis=0)
		self.Y=np.sin(2*np.pi*self.X)+np.random.normal(self.mu,self.sigma,size=(self.set_size,self.set_volume))
		
		
		
		
	def displayer(self):
		fig=plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		#each scatter use a different color to better visulize the dataset
		cmap=plt.cm.get_cmap('jet',self.set_size)
		for i in range(self.set_size):
			xs=self.X[i]
			ys=i
			zs=self.Y[i]
			ax.scatter(xs,ys,zs,c=cmap(i))
		
		ax.set_xlabel('Xlabel')
		ax.set_ylabel('dataset index')
		ax.set_zlabel('Ylabel')
		ax.set_title("synthetic dataset")
		plt.show()


	
	
		

		
#regression model class
class linear_regression():
	
#	
	data=None
#bassis could be power function,sinus,gaussian.	
	basis=None
	basis_num=None
#weight decay regularizer lambda
	lam=None
#mean vector of the Gaussian basis	
	mu=None
#sigma of the Gaussian basis
	sigma=None
#weight vector 	
	weight=None
#designe matrix
	Phi=None
#fitted target	
	Y_predi=None
#psquare rediction bias 
	bias=None
#prediction variance	
	variance=None
#sum of two 
	bias_variance_sum=None	
	
	
	def __init__(self,data=None,basis=None,basis_num=None,lam=None,weight=None,
			  mu=None,sigma=None,Phi=None,Y_predi=None,bias=None,
			  variance=None,bias_variance_sum=None):
		self.data=data
		self.basis=basis
		self.basis_num=basis_num
		self.lam=lam
		self.mu=mu
		self.sigma=sigma
		self.Phi=Phi
		self.weight=weight
		self.Y_predi=Y_predi
		self.bias=bias
		self.variance=variance
		self.bias_variance_sum=bias_variance_sum
		print("class regression model constructed !\n")
	
	def linear_fitting(self):
	#this function shows the fitting result of all the given dataset 	

		if (self.basis=="Gaussian"):
		
		#build the mean tensor of the Gaussian basis
		#first creat a row vector
			mu=np.linspace(0,1,num=self.data.set_volume)
		# repeat the row vector in another dimension	
			mu=np.expand_dims(mu,axis=0)
			mu=np.repeat(mu,self.data.set_volume,axis=0)
		#repeat the matrix in another dimension	
			mu=np.expand_dims(mu,axis=0)
			mu=np.repeat(mu,self.data.set_size,axis=0)
			self.mu=mu
			#print("the shape of mu is now {}\n".format(mu.shape))
		#build the 	sample vector 
		#first we proform dimension expansion
			X=np.expand_dims(self.data.X,axis=1)
		#then do the repeat, depend on the basis number 
			X=np.repeat(X,self.basis_num,axis=1)
		#do the transpose
			X=np.transpose(X,axes=(0,2,1))
			#print("the shape of X is now {}\n".format(X.shape))
		#program the design tensor (the first axis is an matrix called design matrix)
			Phi=np.exp(-np.power(X-mu,2)/(2*np.power(self.sigma,2)))
			#print("the shape of the design tensor Phi is {}\n".format(Phi.shape))
		#program the target matrix
			T=np.transpose(self.data.Y,(1,0))
			T=np.expand_dims(T,axis=2)
			T=np.transpose(T,axes=(1,0,2))
			#print("the shape of the target matrix T is now {}\n".format(T.shape))
		#program the weights matrix
		#first prepare the identity tensor
			I=np.eye(self.basis_num)
			I=np.expand_dims(I,axis=0)
			I=np.repeat(I,self.data.set_size,axis=0)
			#print("the shape of the identity tensor I is {}\n".format(I.shape))
		#program lambdaI+Phi^{T}Phi
			weight=self.lam*I+np.matmul(np.transpose(Phi,axes=(0,2,1)),Phi)
			#print("the shape of the fitted curve weight matrix is {}\n".format(weight.shape))
		#inverse 	
			weight=inv(weight)
			#print("the shape of the fitted curve invert weight matrix is {}\n".format(weight.shape))
		#multiply by Phi^{T}	
			weight=np.matmul(weight,np.transpose(Phi,axes=(0,2,1)))
			#print("the shape of the fitted curve weight multiply by Phi transpose matrix is {}\n".format(weight.shape))
		#multiply by T	
			weight=np.matmul(weight,T)
			#print("the shape of the fitted curve weight matrix is {}\n".format(weight.shape))
			self.weight=weight
		
	def display_basis(self):
		if(self.basis=="Gaussian"):
		#build the mean tensor of the Gaussian basis
		#first creat a row vector
			mu=np.linspace(0,1,num=self.data.set_volume)
		# repeat the row vector in another dimension	
			mu=np.expand_dims(mu,axis=0)
			mu=np.repeat(mu,self.data.set_volume,axis=0)
		#repeat the matrix in another dimension	
			mu=np.expand_dims(mu,axis=0)
			mu=np.repeat(mu,self.data.set_size,axis=0)
			self.mu=mu
			#print("the shape of mu is now {}\n".format(mu.shape))
		#build the 	sample vector 
		#first we proform dimension expansion
			X=np.expand_dims(self.data.X,axis=1)
		#then do the repeat, depend on the basis number 
			X=np.repeat(X,self.basis_num,axis=1)
		#do the transpose
			X=np.transpose(X,axes=(0,2,1))
			#print("the shape of X is now {}\n".format(X.shape))
		#program the design tensor (the first axis is an matrix called design matrix)
			Phi=np.exp(-np.power(X-mu,2)/(2*self.sigma^2))
			self.Phi=Phi
			#print("the shape of the design tensor Phi is {}\n".format(Phi.shape))
			plt.figure()
			for i in range(Phi.shape[1]):
				plt.plot(X[0,0,:],Phi[0,i,:]) 
			
	def display_fitting(self):
		
		#build the 	sample vector 
		#first we proform dimension expansion
			X=np.linspace(0,1,self.data.set_volume)
			X=np.expand_dims(X,axis=0)
			X=np.repeat(X,self.basis_num,axis=0)	
		#then do the repeat, depend on the basis number 
			X=np.expand_dims(X,axis=0)
			X=np.repeat(X,self.data.set_size,axis=0)
			

			
		#do the transpose
			X=np.transpose(X,axes=(0,2,1))
			#print("the shape of X is now {}\n".format(X.shape))
		#program the design tensor (the first axis is an matrix called design matrix)
			Phi=np.exp(-np.power(X-self.mu,2)/(2*np.power(self.sigma,2)))
			#print("the shape of the design tensor Phi is {}\n".format(Phi.shape))
			self.Phi=Phi
		#program the predictions(Y is of shape(100,25,1))
			Y=np.matmul(Phi,self.weight)
			self.Y_predi=Y
			#print("the shape of prediction is {}\n".format(Y.shape))
		#showing all the prediction
			#fig=plt.figure()
			fig, (ax, bx) = plt.subplots(nrows=1, ncols=2, figsize=(9,4), sharey=True)
			#ax = fig.add_subplot(121)

			ax.set_title("\n".join(wrap("regression result of {} dataset using {} basis with the lambda value {}",30)).format(self.data.set_volume,self.basis,self.lam))
			ax.set_xlabel("Xlabel")
			ax.set_ylabel("Ylabel")
			
			cmap=plt.cm.get_cmap('jet',self.data.set_size)
			for i in range(self.data.set_size):

				ax.plot(np.linspace(0,1,self.data.set_volume),np.squeeze(Y[i,:,:]),c=cmap(i))
			

			bx.set_title("\n".join(wrap("the average fitting curve trained by 25 dataset and reference",30)))
			bx.plot(np.linspace(0,1,self.data.set_volume),np.squeeze(np.mean(Y,axis=0)),c="red",label="average curve")
			bx.plot(np.linspace(0,1,self.data.set_volume),np.sin(2*np.pi*np.linspace(0,1,self.data.set_volume)),label="reference curve")
	
			bx.set_xlabel("Xlabel")
			bx.set_ylabel("Ylabel")
			bx.legend()
			print("the variance of the regression result of {} dataset using {} basis with the lambda value {} is {}\n".format(self.data.set_size,self.basis,self.lam,np.mean(np.var(Y,axis=0))))

			
	def bias_variance_evolution(self):
				
		#build the 	sample vector 
		#first we proform dimension expansion
			X=np.linspace(0,1,self.data.set_volume)
			X=np.expand_dims(X,axis=0)
			X=np.repeat(X,self.basis_num,axis=0)	
		#then do the repeat, depend on the basis number 
			X=np.expand_dims(X,axis=0)
			X=np.repeat(X,self.data.set_size,axis=0)
			
		#do the transpose
			X=np.transpose(X,axes=(0,2,1))
			#print("the shape of X is now {}\n".format(X.shape))
		#program the design tensor (the first axis is an matrix called design matrix)
			Phi=np.exp(-np.power(X-self.mu,2)/(2*np.power(self.sigma,2)))
			#print("the shape of the design tensor Phi is {}\n".format(Phi.shape))
			#self.Phi=Phi
		#program the predictions(Y is of shape(100,25,1))
			Y=np.matmul(Phi,self.weight)
			#save the average and the variance and the sum of two term 
			self.bias=np.power(np.squeeze(np.mean(Y,axis=0))-np.sin(2*np.pi*np.linspace(0,1,self.data.set_volume)),2)
			self.bias=np.mean(self.bias)
			self.variance=np.mean(np.var(Y,axis=0))
			self.bias_variance_sum=self.bias+self.variance


#------------------------------program part-----------------------------------#

#generation of 100 dataset each contains 25 data

data=Data(set_size=100,set_volume=25,mu=0,sigma=0.4)
data.generator()

#show all the generative data set in 3D Plot

data.displayer()

#---------------------------simple visulization-------------------------------# 

#creat a model_0
model_0=linear_regression(data=data,basis="Gaussian",basis_num=25,lam=10,sigma=0.3)
#training
model_0.linear_fitting()
#model.display_basis()
model_0.display_fitting()

#creat a model_1
model_1=linear_regression(data=data,basis="Gaussian",basis_num=25,lam=1,sigma=0.3)
#training
model_1.linear_fitting()
#model.display_basis()
model_1.display_fitting()

#creat a model_2
model_2=linear_regression(data=data,basis="Gaussian",basis_num=25,lam=0.1,sigma=0.3)
#training
model_2.linear_fitting()
#model.display_basis()
model_2.display_fitting()

#%%
#creat a bias and variance evoution in function of the model complexity gouverned by lamda
lam=np.exp(np.linspace(-3,3,50))
plt.figure()
for i in range(50):

	model=linear_regression(data=data,basis="Gaussian",basis_num=25,lam=lam[i],sigma=0.1)
	model.linear_fitting()
	model.bias_variance_evolution()
	if(i==49):
		plt.plot(np.log(lam[i]),model.bias,"rx",label="square bias")
		plt.plot(np.log(lam[i]),model.variance,"bx",label="variance")
		plt.plot(np.log(lam[i]),model.bias_variance_sum,"gx",label="sum of the square bias and variance")
	else:
		plt.plot(np.log(lam[i]),model.bias,"rx")
		plt.plot(np.log(lam[i]),model.variance,"bx")
		plt.plot(np.log(lam[i]),model.bias_variance_sum,"gx")

plt.xlabel("ln(lambda)")
plt.ylabel("value")
plt.legend()	






