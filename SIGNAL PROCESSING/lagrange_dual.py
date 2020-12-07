#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 22:30:58 2020

@author: shengkailin
"""

import numpy as np
#用拉格朗日对偶法来解本问题
#首先求使得w使得L(w,alpha,lambda,phi)最小的关系w(alpha,lambda,phi),然后把L中w的部分代换
#得到一个仅依赖 alpha，lambda，phi的函数g，找到最大化这个函数g的极点然后得到w的值

def Lagrange_multiplier(w,alpha,beta,gamma,A,b):
	"""
	
本问题是一个带等式和不等式约束的优化问题，该问题的标准型式为
minf(x) 
s.t
g_i(w)<=0,i=1,2,3...m
h_i(w)-beta_i=0,i=1,2,3...m
其中alpha代表涉及不等关系的那些方程的拉格朗日乘数，根据kkt条件这些项必须非负
其中beta代表涉及归一化资产那些方程的拉格朗日乘数
gamma代表的是涉及归一化负债那些拉格朗日乘数
A，b代表了约束Aw-b=0

	"""
	return sum(w*np.log(w))-alpha.dot(w)+(A.dot(w)-b).dot(np.concatenate((beta,gamma),axis=0))

def gradient_Lagrange_min(alpha,beta,gamma,A,b):
	"""
	该方程里已用x的最小值函数代替x因而拉格朗日函数转换成函数theta仅仅依赖alpha，beta，gamma
	w_ij=exp(alpha_i-beta_i-gamma_j-1)
	梯度是函数theta对alpha，beta，gamma
	
	"""
	#gradient=np.zeros(len(alphalen(beta)+len(gamma))
	dalpha=np.zeros(25)
	dbeta=np.zeros(5)
	dgamma=np.zeros(5)
	
	w=np.zeros((5,5))
	for i in range(5):
		for j in range(5):
			w[i][j]=np.exp(alpha[i]-beta[i]-gamma[j]-1)
	h=1e-3
	w=np.reshape(w,(25))
	#print(w)		
	for i in range(25):
		step=np.zeros(25)
		step[i]=h
		dl=Lagrange_multiplier(w,alpha+step,beta,gamma,A,b)-Lagrange_multiplier(w,alpha,beta,gamma,A,b)/h
		dalpha[i]=dl
	
	for i in range(5):
		step=np.zeros(5)
		step[i]=h
		dl=Lagrange_multiplier(w,alpha,beta+step,gamma,A,b)-Lagrange_multiplier(w,alpha,beta,gamma,A,b)/h
		dbeta[i]=dl
	for i in range(5):
		step=np.zeros(5)
		step[i]=h
		dl=Lagrange_multiplier(w,alpha,beta,gamma+step,A,b)-Lagrange_multiplier(w,alpha,beta,gamma,A,b)/h
		dgamma[i]=dl	
	gradient=np.concatenate((dalpha,dbeta),axis=0)
	gradient=np.concatenate((gradient,dgamma),axis=0)
	return gradient,Lagrange_multiplier(w,alpha,beta,gamma,A,b)

alpha=np.ones(25)
beta=np.ones(5)
gamma=np.ones(5)
total=np.ones(35)
lr=1e-2
error0=0
epsilon=1e-9
cnt=0
A=np.array([[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
			[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
			[0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],
			[0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],
			[0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],
			[0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]])
#b向量是a向量与l向量在列方向上的合并
#归一化资产：资产总和为1
a=np.array([26289,13463,732152,7528,28382])
a=a/np.sum(a)
#归一化负债:负债综和为1
l=np.array([0,207017,255588,329515,15694])
l=l/np.sum(l)
b=np.concatenate((a,l),axis=0)
while True:
	cnt+=1
	#设置手动进行循环
	input("输入回车键继续...")
	#计算梯度和拉格朗日下界theta函数的值
	gradient,upper_bound=gradient_Lagrange_min(alpha,beta,gamma,A,b)
	#梯度的归一化
	gradient=gradient/np.linalg.norm(gradient)
	#根据对偶问题的特性，这里应该求使theta函数达到极大的点
	total=total+lr*gradient
	total[:25]=np.max(total[:25],0)
	#更新参数
	alpha=total[:25]
	beta=total[25:30]
	gamma=total[30:35]
	#停止的条件
	error1=upper_bound
	if(abs(error1-error0)<epsilon):
		break
	else:
		error0=error1
	print("最新参数为{},熵为{}".format(total,error0))
	
