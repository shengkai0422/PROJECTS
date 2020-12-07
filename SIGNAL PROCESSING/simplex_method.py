#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:42:15 2020

@author: shengkailin
"""



import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
#要估计的双边联系矩阵的纬度
p=5
#构造归一化矩阵元素w的范围:0 到1之间
bounds=Bounds(np.zeros(p**2),np.ones(p**2))
#等式关系
#归一化资产：资产端总和为1
a=np.array([26289,13463,732152,7528,28382])
a=a/np.sum(a)
#归一化负债:负债端总和为1
l=np.array([0,207017,255588,329515,15694])
l=l/np.sum(l)
#求熵函数的最大值就是求负熵的最小值
def Neg_entropy(x):
	return sum(x*np.log(x))
#负熵函数的雅可比矩阵
def Der_negentropy(x):
	
	der=np.zeros_like(x)
	for i in range (len(x)):
		der[i]=x[i]+1
	return der	
	
#梯度平方和函数
def gradient_square(x):
	r=1000
	alpha=x[25:30]
	beta=x[30:35]
	#归一化资产：资产端总和为1
	a=np.array([26289,13463,732152,7528,28382])
	a=a/np.sum(a)
	#归一化负债:负债端总和为1
	l=np.array([0,207017,255588,329515,15694])
	l=l/np.sum(l)
	
	w=np.reshape(x[:25],(5,5))
	sum_=0
	for i in range(5):
		for j in range(5):
			sum_=sum_+(np.log(w[i][j])+alpha[i]+beta[j])**2
			
	sum_=sum_+(x[0]+x[1]+x[2]+x[3]+x[4]-a[0])**2+\
		(x[5]+x[6]+x[7]+x[8]+x[9]-a[1])**2+\
		(x[10]+x[11]+x[12]+x[13]+x[14]-a[2])**2+\
		(x[15]+x[16]+x[17]+x[18]+x[19]-a[3])**2+\
		(x[20]+x[21]+x[22]+x[23]+x[24]-a[4])**2+\
		(x[0]+x[5]+x[10]+x[15]+x[20]-l[0])**2+\
		(x[1]+x[6]+x[11]+x[16]+x[21]-l[1])**2+\
		(x[2]+x[7]+x[12]+x[17]+x[22]-l[2])**2+\
		(x[3]+x[8]+x[13]+x[18]+x[23]-l[3])**2+\
		(x[4]+x[9]+x[14]+x[19]+x[24]-l[4])**2	
		
	return sum_

x0=np.ones(35)*0.3
res = minimize(gradient_square, x0, method='nelder-mead',
				options={'xatol': 1e-8, 'disp': True})
print(np.reshape(res.x[:25],(5,5)))
#理论值
theory_value=np.zeros((5,5))
#赋值
for i in range(len(a)):
	for j in range(len(l)):
		theory_value[i][j]=a[i]*l[j]
print(theory_value)
print(matrix-theory_value)	

	
			
			