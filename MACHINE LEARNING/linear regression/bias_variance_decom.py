#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:59:33 2020

@author: shengkailin
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
#-----------------------------------------------------------------------------#

#------------------------------libraries--------------------------------------#
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from matplotlib import pyplot as plt
from scipy import stats
from textwrap import wrap
#-----------------------------------------------------------------------------#


#--------------------------------classes--------------------------------------#
class Dataset_generator():
	"""
	This is a data generator for regression reseach，it can generate N 1D datasets
	of length M. These dataset can be added some noises. The train and test are 
	of proportion 3:1.
	
	Attributes:
		X_train: training data 2D matrix
		Y_train: training label 2D matrix 
		X_test: test data 2D matrix
		Y_test: test label 2D matrix 
		Dataset_wid: how many datasets
		Dataset_len: the length of each dataset
		train_len: the length of the train set (75% of the dataset)
		test_len:the length of th test set (25% of the dataset)
	
	"""

	#how many dataset are defined 
	Dataset_wid=None
	#in each dataset how many samples 
	Dataset_len=None
	#type of noise :Gaussian,Poisson,
	noise_type=None
	#variance of the noise
	noise_var=None
	
	def __init__(self,Dataset_wid=None,Dataset_len=None,noise_type=None,noise_var=None):
		
		#define basic variable
		self.Dataset_wid=Dataset_wid
		self.Dataset_len=Dataset_len
		self.noise_type=noise_type
		self.noise_var=noise_var
		self.train_len=int(0.75*self.Dataset_len)
		self.test_len=int(0.25*self.Dataset_len)
		#first compose a input vector 
		X=np.linspace(0,1,self.Dataset_len)
		#random sample and no repititio sampling(replace parameter)
		X_train=np.random.choice(X,size=self.train_len,replace=False)
		X_test=np.random.choice(X,size=self.test_len,replace=False)
		
		#repeat Dataset_len times
		self.X_train=np.repeat(np.expand_dims(X_train,axis=0),self.Dataset_wid,axis=0)
		self.X_test=np.repeat(np.expand_dims(X_test,axis=0),self.Dataset_wid,axis=0)
		while True:
			 try:
				 
				 #we suppose the p(t|x) density is unimodal
				 if(self.noise_type=="Gaussian"):
					 
					  #label=sinus+noise
					 self.Y_train=np.sin(2*np.pi*self.X_train)\
						 +np.random.normal(0,self.noise_var,size=(self.Dataset_wid,self.train_len))
					 
					 self.Y_test=np.sin(2*np.pi*self.X_test)\
						 +np.random.normal(0,self.noise_var,size=(self.Dataset_wid,self.test_len)) 
				 
					 break
				 
				 elif(self.noise_type=="Poisson"):

					 break
				 
				 elif(self.noise_type=="Uniform"):
					 break
					 
				 elif(self.noise_type=="Gamma"):
					  break
				 else:
					 #raise an error to retype the noise type
					 raise
				 
					 
					 
			 except RuntimeError:
				 print("The noise type is invalid!\n")
				 self.noise_type=input("please enter a valid noise type:")
	
	
	def display_trainset(self):
		fig=plt.figure()
		#add an axis object ax1 to display the point(x,i,y) with i being the
		#dataset index (first,second ,third ....)
		ax1=fig.add_subplot(111,projection="3d")
		#each scatter use a different color to better visulize the dataset
		#cmap=plt.cm.get_cmap('jet',self.Dataset_wid)
		cmap=plt.cm.jet(np.linspace(0,1,self.Dataset_wid))
		#setup the axis
		ax1.set_xlabel('Xlabel')
		ax1.set_ylabel('how many dataset')
		ax1.set_zlabel('Ylabel')
		ax1.set_title("synthetic dataset")
		#plot the scatter of all train set 
		for idx in range(self.Dataset_wid):

			ax1.scatter(self.X_train[idx],idx,self.Y_train[idx],c=np.expand_dims(cmap[idx],axis=0))
		

		
	def display_trainset_histo(self):
		#This ax is for the histogram
		fig=plt.figure()
		ax2=fig.add_subplot(111,projection="3d")
		camp=plt.cm.jet(np.linspace(0,1,5))
		#plot all the distribution p(y|x), the length of x is the length 
		#of the training set
		for idx in range(3):
			
			hist,bins=np.histogram(self.Y_train[:,idx],bins=15,density=True)
			xs=(bins[:-1]+bins[1:])/2
			#zs gives Xlabel zidr='y' means the histogram is in the plan x-z,
			#alpha gives the transparency,width give the binwidth 
			ax2.bar(xs,hist,zs=self.X_train[0,idx],zdir='y',color=camp[idx],
		   edgecolor=camp[idx],linewidth=3,alpha=0.8,width=0.1)
			#set up the axis 
			ax2.set_xlabel('Ylabel')
			ax2.set_ylabel('Xlabel')
			ax2.set_zlabel('Probability p(y|x)')
			ax2.set_title('Training set histogram and density fitting')
			#fit to its own distribution:Gaussian,Poisson,Uniform,etc
			if(self.noise_type=="Gaussian"):
				#inference the mean and standart derivation(biased estimator)
				mu,std = stats.norm.fit(self.Y_train[:,idx])
				print("fitting the density p(y|x={}) to Gaussian one:\n \
		  the mean is {} the variance is {}\n".format(self.X_train[0,idx],mu,std))
				#set up the density range 
				ymin=np.min(xs)
				ymax=np.max(xs)
				y=np.linspace(ymin,ymax,self.Dataset_wid)
				#generate density
				pdf_norm = stats.norm.pdf(y, mu,std)
				#add to plot,attenstion in the y direction it should be a vector 
				#so we use np.repeat
				ax2.plot(y,np.repeat(self.X_train[0,idx],self.Dataset_wid),pdf_norm,linewidth=4,color=camp[idx])
			
			elif(self.noise_type=="Poisson"):
				break
			elif(self.noise_type=="Uniform"):
				break
			#this is the Gamma case
			else:
				break

#------------------------function used in the class--------------------------#
def gaussian_basis(x,y,width):
	arg=(x-y)/width 
	return np.exp(-0.5*(arg**2))
#-----------------------------------------------------------------------------#
class Linear_regression():
	"""
	This model class has several fonctions:
		1.choose the basis and visulize basis
		2.feature enginnering : preprocess the data into another feature space
		3.train :training the model , getting the parameter w, train error
		4.test :getting the test error
		5. see the regession result of all dataset, compute mean 
		and variance of the results.
		6.plot train error, test error, bias,variance, bias+variance with 
		respect to the  lambda
		

	"""
	data=None
	basis=None
	basis_num=None
	lam=None
	std=None
	def	__init__(self,data=None,basis=None,basis_num=None,lam=None,std=None):
		self.data=data
		self.basis=basis
		self.basis_num=basis_num
		self.lam=lam
		self.basis_var=std
	
	def basis_display(self):
		#new figure for basis
		fig=plt.figure()
		ax1=fig.add_subplot(111)
		#setup the axis
		ax1.set_xlabel('Xlabel')
		ax1.set_ylabel('Ylabel')
		if(self.basis=="Gaussian"):
			#here must do a type conversion//
			#we abandone this way because its not easy to give the evolution
			#bias-variance after
			#self.basis_var=float(input("give the gaussian basis std:"))
		
			#generate gaussian basis function cenetred at each Xtrain label 
			for idx in range(self.data.train_len):
				
				y=gaussian_basis(np.linspace(0,1,100),np.repeat(self.data.X_train[0,idx],100),self.basis_var)
				#this if else is for input legend 
				if(idx==0):
					ax1.plot(np.linspace(0,1,100),y,color="red",label="gaussian basis function")				
				else:
					ax1.plot(np.linspace(0,1,100),y,color="red")
			ax1.scatter(self.data.X_train[0],self.data.Y_train[0],color="blue",label="data")	
			ax1.legend()
			ax1.set_title("\n".join(wrap("{} bassis , std is {}and data ".format(self.basis,self.basis_var),30)))

	def feature_preprocess(self,mode=None):
		if (self.basis=="Gaussian"):
		#build the mean tensor of the Gaussian basis
		#first creat a row vector
			mu=np.linspace(0,1,num=self.basis_num)
		# repeat the row vector in another dimension	
			mu=np.expand_dims(mu,axis=0)
			if(mode=="train"):
				mu=np.repeat(mu,self.data.train_len,axis=0)
				#repeat the matrix in another dimension	
				mu=np.expand_dims(mu,axis=0)
				mu=np.repeat(mu,self.data.Dataset_wid,axis=0)
				self.mu_train=mu
				#build the 	sample vector 
				#first we proform dimension expansion
				X=np.expand_dims(self.data.X_train,axis=1)
				#then do the repeat, depend on the basis number 
				X=np.repeat(X,self.basis_num,axis=1)
				#do the transpose
				X=np.transpose(X,axes=(0,2,1))
				#program the design tensor (the first axis is an matrix called design matrix)
				self.phi=np.exp(-np.power(X-mu,2)/(2*np.power(self.basis_var,2)))
			else:
				mu=np.repeat(mu,self.data.test_len,axis=0)
				#repeat the matrix in another dimension	
				mu=np.expand_dims(mu,axis=0)
				mu=np.repeat(mu,self.data.Dataset_wid,axis=0)
				self.mu_test=mu
				#build the 	sample vector 
				#first we proform dimension expansion
				X=np.expand_dims(self.data.X_test,axis=1)
				#then do the repeat, depend on the basis number 
				X=np.repeat(X,self.basis_num,axis=1)
				#do the transpose
				X=np.transpose(X,axes=(0,2,1))
				#print("the shape of X is now {}\n".format(X.shape))
				#program the design tensor (the first axis is an matrix called design matrix)
				self.phi=np.exp(-np.power(X-mu,2)/(2*np.power(self.basis_var,2)))
		
	
	def train(self):
		#program the target matrix T
		T=np.transpose(self.data.Y_train,(1,0))
		T=np.expand_dims(T,axis=2)
		T=np.transpose(T,axes=(1,0,2))
		#print("the shape of the target matrix T is now {}\n".format(T.shape))
		#program the weights matrix
		#first prepare the identity tensor
		I=np.eye(self.basis_num)
		I=np.expand_dims(I,axis=0)
		I=np.repeat(I,self.data.Dataset_wid,axis=0)
		#print("the shape of the identity tensor I is {}\n".format(I.shape))
		#program lambdaI+Phi^{T}Phi
		weight=self.lam*I+np.matmul(np.transpose(self.phi,axes=(0,2,1)),self.phi)
		#print("the shape of the fitted curve weight matrix is {}\n".format(weight.shape))
		#inverse 	
		weight=inv(weight)
		#print("the shape of the fitted curve invert weight matrix is {}\n".format(weight.shape))
		#multiply by Phi^{T}	
		weight=np.matmul(weight,np.transpose(self.phi,axes=(0,2,1)))
		#print("the shape of the fitted curve weight multiply by Phi transpose matrix is {}\n".format(weight.shape))
		#multiply by T	
		weight=np.matmul(weight,T)
		#print("the shape of the fitted curve weight matrix is {}\n".format(weight.shape))
		self.weight=weight
		#prediction curve (regression result using training data)
		self.Y_predi=np.matmul(self.phi,self.weight)
		# mean training error (quadratic)
		self.train_error=np.mean(np.power(np.squeeze(self.Y_predi)-self.data.Y_train,2))

	
	def train_display(self):
		
		#build the 	sample vector, this vector must be an increasing order 
		#becasue we plot method requires so 
		#first we proform dimension expansion
			X=np.linspace(0,1,self.data.train_len)
			X=np.expand_dims(X,axis=0)
			X=np.repeat(X,self.basis_num,axis=0)	
		#then do the repeat, depend on the basis number 
			X=np.expand_dims(X,axis=0)
			X=np.repeat(X,self.data.Dataset_wid,axis=0)	
		#do the transpose
			X=np.transpose(X,axes=(0,2,1))
			#print("the shape of X is now {}\n".format(X.shape))
		#program the design tensor (the first axis is an matrix called design matrix)
			Phi=np.exp(-np.power(X-self.mu_train,2)/(2*np.power(self.basis_var,2)))
			#print("the shape of the design tensor Phi is {}\n".format(Phi.shape))
		#program the predictions(Y is of shape(100,25,1))
			Y=np.matmul(Phi,self.weight)
			#print("the shape of prediction is {}\n".format(Y.shape))
		#showing all the prediction curves 
			fig, (ax, bx) = plt.subplots(nrows=1, ncols=2, figsize=(9,4), sharey=True)
			ax.set_title("\n".join(wrap("regression result of {} dataset using {} basis with the lambda value {}",30)).format(self.data.Dataset_wid,self.basis,self.lam))
			ax.set_xlabel("Xlabel")
			ax.set_ylabel("Ylabel")
			
			cmap=plt.cm.get_cmap('jet',self.data.Dataset_wid)
			for i in range(self.data.Dataset_wid):

				ax.plot(np.linspace(0,1,self.data.train_len),np.squeeze(Y[i,:,:]),c=cmap(i))
			

			bx.set_title("\n".join(wrap("the average fitting curve trained by 25 dataset and reference",30)))
			bx.plot(np.linspace(0,1,self.data.train_len),np.squeeze(np.mean(Y,axis=0)),c="red",label="average curve")
			bx.plot(np.linspace(0,1,self.data.train_len),np.sin(2*np.pi*np.linspace(0,1,self.data.train_len)),label="reference curve")
	
			bx.set_xlabel("Xlabel")
			bx.set_ylabel("Ylabel")
			bx.legend()
			print("the variance of the regression result of {} dataset using {} basis with the lambda value {} is {}\n".format(self.data.Dataset_wid,self.basis,self.lam,np.mean(np.var(Y,axis=0))))

	def test(self):
		self.feature_preprocess(mode="test") 
		self.Y_predi_test=np.matmul(self.phi,self.weight)
		#mean expected test loss
		self.test_error=np.mean(np.power(np.squeeze(self.Y_predi_test)-self.data.Y_test,2))
	
	def bias_variance(self):
		#square bias
		self.bias=np.mean(np.power(np.squeeze(np.mean(self.Y_predi,axis=0))-
		 np.sin(2*np.pi*self.data.X_train[0]),2))
		# variance 			
		self.variance=np.mean(np.var(self.Y_predi,axis=0))
		#bias_variance_sum
		self.bias_variance_sum=self.bias+self.variance

#creat a bias and variance evoution in function of the model complexity gouverned by lamda
def bias_variance_evolution(data):
	lam=np.exp(np.linspace(-3,3,50))
	plt.figure()
	for i in range(50):
		model=Linear_regression(data=data,basis="Gaussian",basis_num=25,lam=lam[i],std=0.1)
		model.feature_preprocess(mode="train")
		model.train()
		model.test()
		model.bias_variance()
		if(i==49):
			plt.plot(np.log(lam[i]),model.train_error,"bx",label="train error")
			plt.plot(np.log(lam[i]),model.test_error,"x",color="orange",label="test error")
			plt.plot(np.log(lam[i]),model.bias,"rx",label="square bias")
			plt.plot(np.log(lam[i]),model.variance,"x",color="black",label="variance")
			plt.plot(np.log(lam[i]),model.bias_variance_sum,"gx",label="sum of the square bias and variance")
		else:
			plt.plot(np.log(lam[i]),model.train_error,"bx")
			plt.plot(np.log(lam[i]),model.test_error,"x",color="orange")
			plt.plot(np.log(lam[i]),model.bias,"rx")
			plt.plot(np.log(lam[i]),model.variance,"x",color="black")
			plt.plot(np.log(lam[i]),model.bias_variance_sum,"gx")

	plt.xlabel("ln(lambda)")
	plt.ylabel("value")
	plt.legend()	
		
		
	
		
		
		

	
		
		
				
		
		
				

		
		 
					  
			
#-----------------------------------------------------------------------------

#----------------------------------program------------------------------------#
#dataset explore 
data=Dataset_generator(Dataset_wid=50,Dataset_len=40,noise_type="Gaussian",noise_var=0.3)		
data.display_trainset()
data.display_trainset_histo()
#%%	
model=Linear_regression(data=data,basis="Gaussian",basis_num=10,lam=1,std=0.2)
model.basis_display()
#%%
model.feature_preprocess(mode="train")
model.train()
#%%
model.train_display()
#%%
model.test()
#%%
bias_variance_evolution(data=data)


		
	
	

