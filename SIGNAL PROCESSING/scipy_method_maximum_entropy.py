#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:26:14 2020

@author: shengkailin
"""
#使用的库
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize

#自定义函数
#求熵函数的最大值就是求负熵的最小值
def Neg_entropy(x):
	return sum(x*np.log(x))
#负熵函数的雅可比矩阵
def Der_negentropy(x):
	
	der=np.zeros_like(x)
	for i in range (len(x)):
		der[i]=x[i]+1
	return der	
#理论上的结果,根据我们的数学推导得到的
def theory_value(a,l):
	theory_value=np.zeros((5,5))
	for i in range(len(a)):
		for j in range(len(l)):
			theory_value[i][j]=a[i]*l[j]

	return theory_value
#一些基本量的定义
#要估计的双边联系矩阵的维度
p=5
#构造归一化矩阵元素w的范围:0 到1之间
bounds=Bounds(np.zeros(p**2),np.ones(p**2))
#等式关系
#总资产
a=np.array([26289,13463,732152,7528,28382])
#归一化总资产
a=a/np.sum(a)
#总负债
l=np.array([0,207017,255588,329515,15694])
#归一化总负债
l=l/np.sum(l)	
# 等式约束条件：fun是等式条件的具体形式，jac表示的是等式约束的雅可比矩阵
eq_cons = {'type': 'eq',
		'fun' : lambda x: np.array([x[0]+x[1]+x[2]+x[3]+x[4]-a[0],
								x[5]+x[6]+x[7]+x[8]+x[9]-a[1],
								x[10]+x[11]+x[12]+x[13]+x[14]-a[2],
								x[15]+x[16]+x[17]+x[18]+x[19]-a[3],
								x[20]+x[21]+x[22]+x[23]+x[24]-a[4],
								x[0]+x[5]+x[10]+x[15]+x[20]-l[0],
								x[1]+x[6]+x[11]+x[16]+x[21]-l[1],
								x[2]+x[7]+x[12]+x[17]+x[22]-l[2],
								x[3]+x[8]+x[13]+x[18]+x[23]-l[3],
								x[4]+x[9]+x[14]+x[19]+x[24]-l[4]]),
		'jac' : lambda x: np.array([[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
								[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
								[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
								[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
								[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
								[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
								[0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],
								[0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],
								[0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],
								[0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]])}
#不等式约束条件，fun、表示的是标准大于等于不等式约束 jac表示的是不等式约束的雅可比矩阵
ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],
										 x[8],x[9],x[10],x[11],x[12],x[13],x[14],
										 x[15],x[16],x[17],x[18],x[19],x[20],
										 x[21],x[22],x[23],x[24]]),
             'jac' : lambda x:np.eye(25)}

#设定初值
x0 = np.ones(p**2)*0.1
#使用时序最小二乘法Sequential Least Square programming
res = minimize(Neg_entropy, x0, method='SLSQP', jac=Der_negentropy,
			   constraints=[eq_cons,ineq_cons], options={'ftol': 1e-11, 'disp': True},
			   bounds=bounds)
matrix=np.reshape(res.x,(p,p))
print(matrix)
#理论值
value=theory_value(a,l)
#比较两者差异
print((matrix-value))		
##-----