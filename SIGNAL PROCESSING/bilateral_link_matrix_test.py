#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:23:11 2020

@author: shengkailin
"""
import numpy as np
import math
from scipy.optimize import minimize
#熵的定义是 -xlog（x）
def Entropy(x):
	return sum(-x*np.log(x))
#定义一个非常小的数
e=1e-9
value=np.array([17440.09,e,1665869,e,55772.83,546240.1,817730.9,e,312376.2,89984.55])
#对于非合并的资产负债表，双边联系矩阵对角线上的元素可以为0，因此使用最大熵估计法
cons = ({'type': 'eq', 'fun': lambda x: x[0]+ x[1]+ x[2]+x[3]+x[4]- value[0]},
{'type': 'eq', 'fun': lambda x: x[5]+ x[6]+ x[7]+x[8]+x[9]-value[1]},
{'type': 'eq', 'fun': lambda x: x[10]+ x[11]+ x[12]+x[13]+x[14] - value[2]},
{'type': 'eq', 'fun': lambda x: x[15]+ x[16]+ x[17]+x[18]+x[19]-value[3]},
{'type': 'eq', 'fun': lambda x: x[20]+ x[21]+ x[22]+x[23]+x[24] - value[4]},
{'type': 'eq', 'fun': lambda x: x[0]+ x[5]+ x[10]+x[15]+x[20] - value[5]},
{'type': 'eq', 'fun': lambda x: x[1]+ x[6]+ x[11]+x[16]+x[21] - value[6]},
{'type': 'eq', 'fun': lambda x: x[2]+ x[7]+ x[12]+x[17]+x[22] - value[7]},
{'type': 'eq', 'fun': lambda x: x[3]+ x[8]+ x[13]+x[18]+x[23] - value[8]},
{'type': 'eq', 'fun': lambda x: x[4]+ x[9]+ x[14]+x[19]+x[24] - value[9]})	

x0=np.ones(25)*100000
fun=Entropy
res = minimize(fun, x0, method='SLSQP', constraints=cons,options={'ftol': 1e-9, 'disp': True})
print('最大值：',res.fun)
print('最优解：',res.x)
print('迭代终止是否成功：', res.success)
print('迭代终止原因：', res.message)


#%%
e = 1e-10 # 非常接近0的值
fun = lambda x : 8 * (x[0] * x[1] * x[2]) # f(x,y,z) =8 *x*y*z
cons = ({'type': 'eq', 'fun': lambda x: x[0]**2+ x[1]**2+ x[2]**2 - 1}, # x^2 + y^2 + z^2=1
        {'type': 'ineq', 'fun': lambda x: x[0] - e}, # x>=e等价于 x > 0
        {'type': 'ineq', 'fun': lambda x: x[1] - e},
        {'type': 'ineq', 'fun': lambda x: x[2] - e}
       )
x0 = np.array((-5, -5, -5)) # 设置初始值
res = minimize(fun, x0, method='SLSQP', constraints=cons,options={'ftol': 1e-9, 'disp': True})


print('最大值：',res.fun)
print('最优解：',res.x)
print('迭代终止是否成功：', res.success)
print('迭代终止原因：', res.message)

#%%另一种思路，先定义带拉格朗日约束条件的完整函数，再定义导数函数，最后求得零点
#问题：当初始点离极大值点很远时，必须使用barrier让矩阵参数保持大于0，否则将会出现错误
#问题：数据本身有问题
#解决方法
#1）最简单的方法是不去演化那些为0的元素
#2）复杂一点的方法是使用barrier方法
def func(x):
	value=1e-5*np.array([17440.09,e,1665869,e,55772.83,546240.1,817730.9,e,312376.2,89984.55])

	return sum(-x[:25]*np.log(x[:25]))+x[25]*(x[0]+ x[1]+ x[2]+x[3]+x[4]- value[0])+\
		x[26]*(x[5]+ x[6]+ x[7]+x[8]+x[9]-value[1])+\
		x[27]*(x[10]+ x[11]+ x[12]+x[13]+x[14] - value[2])+\
		x[28]*(x[15]+ x[16]+ x[17]+x[18]+x[19]-value[3])+\
		x[29]*(x[20]+ x[21]+ x[22]+x[23]+x[24] - value[4])+\
		x[30]*(x[0]+ x[5]+ x[10]+x[15]+x[20] - value[5])+\
		x[31]*(x[1]+ x[6]+ x[11]+x[16]+x[21] - value[6])+\
		x[32]*(x[2]+ x[7]+ x[12]+x[17]+x[22] - value[7])+\
		x[33]*(x[3]+ x[8]+ x[13]+x[18]+x[23] - value[8])+\
		x[34]*(x[4]+ x[9]+ x[14]+x[19]+x[24] - value[9])	

def dfunc(x):
	value=1e-5*np.array([17440.09,e,1665869,e,55772.83,546240.1,817730.9,e,312376.2,89984.55])

	dlambda=np.zeros(len(x))
	h=1e-3
	for i in range (len(x)):
		dx=np.zeros(len(x))
		dx[i]=h
		dlambda[i]=(func(x+dx)-func(x-dx))/(2*h)
	dlambda[]
	return dlambda	

from scipy.optimize import fsolve
value=1e-5*np.array([17440.09,e,1665869,e,55772.83,546240.1,817730.9,e,312376.2,89984.55])
# this is the max
X1 = fsolve(dfunc, np.ones(35))
#学习率，更新率
lr=1e-3
#停止条件
epsilon=1e-3
error0=func(np.ones(35))
x=np.ones(35)
#迭代次数
cnt=0
#print X1, func(X1,value)
while True:
	#设置手动进行循环
	input("输入回车键继续...")
	cnt+=1
	#函数值
	function_value=func(x)
	#梯度
	gradient=dfunc(x)
	#极值点向梯度正方向演化寻找最大值
	x=x+lr*gradient
	
	#停止的条件
	error1=func(x)
	if abs(error1-error0)<epsilon:
		break
	else:
		error0=error1
	print("最新参数为{},熵为{}".format(x,error0))
	
print("迭代终了,极值点是{},函数值为{}".format(x,func(x)))
print(" 迭代次数是{}".format(cnt))		
	

#%%
def func(w):
	value=np.array([17440.09,1665869,55772.83,546240.1,817730.9,312376.2,89984.55])
	return sum(-w[:12]*np.log(w[:12]))+w[13]*(w[0]+w[1]+w[2]+w[3]-value[0])+\
		w[14]*(w[4]+w[5]+w[6]+w[7]-value[1])+\
		w[15]*(w[8]+w[9]+w[10]+w[11]-value[2])+\
		w[16]*(w[0]+w[4]+w[8]-value[3])+\
		w[17]*(w[1]+w[5]+w[9]-value[4])+\
		w[18]*(w[2]+w[6]+w[10]-value[5])+\
		w[19]*(w[3]+w[7]+w[11]-value[6])

def dfunc(w):

	dlambda=np.zeros(len(w))
	h=1e-3
	for i in range (len(w)):
		dw=np.zeros(len(w))
		dw[i]=h
		dlambda[i]=(func(w+dw)-func(w-dw))/(2*h)
	return dlambda		
		
#学习率，更新率
lr=1e-3
#停止条件
epsilon=1e-3
#定初值
from sympy import *
import sympy as sp
a=Matrix([[1,1,1,1,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,1,0,0,0,0],[0,0,0,0,0,
0,0,0,1,1,1,1],[1,0,0,0,1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0,0,1,0,0],[0,0,1,0,0,
0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1,0,0,0,1]])
b=Matrix([[17440.09],[1665869],[55772.83],[546240.1],[817730.9],[312376.2],
[89984.55]])
w=sp.symbols('w')
w=Matrix(sp.symarray(w,12))
t=sp.solve(a*w-b)
#%%
#迭代次数
cnt=0
#print X1, func(X1,value)
while True:
	#设置手动进行循环
	input("输入回车键继续...")
	cnt+=1
	#函数值
	function_value=func(x)
	#梯度
	gradient=dfunc(x)
	#极值点向梯度正方向演化寻找最大值
	x=x+lr*gradient
	
	#停止的条件
	error1=func(x)
	if abs(error1-error0)<epsilon:
		break
	else:
		error0=error1
	print("最新参数为{},熵为{}".format(x,error0))
	
print("迭代终了,极值点是{},函数值为{}".format(x,func(x)))
print(" 迭代次数是{}".format(cnt))		
			
#%%
import numpy as np
#使用拉格朗日乘数加对数障碍函数法来估计双边联系矩阵
#选用拉格朗日乘数法是因为有等式约束条件:每一行元素的和等于资产，每一列元素的和负债
#选用障碍函数法是因为有不等式约束条件 :矩阵内任意元素w_ij>=0
#这里选用的障碍函数为对数函数，其他可能性有反比例函数
#归一化资产：资产总和为1
a=np.array([26289,13463,732152,7528,28382])
a=a/np.sum(a)
#归一化负债:负债综和为1
l=np.array([0,207017,255588,329515,15694])
l=l/np.sum(l)

#障碍法的miu参数
miu=np.array([10,1,0.1])
#梯度递降法的学习率
lr=1e-3
def func(w,a,l,miu):
		#本题是要求函数F=（熵+拉格朗日乘数部分）的最大值，转换成求-F的最小值。
		#进一步使用障碍函数，求（-F-miu*障碍函数和）的最小值
	return sum(w[:25]*np.log(w[:25]))-w[25]*(w[0]+w[1]+w[2]+w[3]+w[4]-a[0])-\
		w[26]*(w[5]+w[6]+w[7]+w[8]+w[9]-a[1])-\
		w[27]*(w[10]+w[11]+w[12]+w[13]+w[14]-a[2])-\
		w[28]*(w[15]+w[16]+w[17]+w[18]+w[19]-a[3])-\
		w[29]*(w[20]+w[21]+w[22]+w[23]+w[24]-a[4])-\
		w[30]*(w[0]+w[5]+w[10]+w[15]+w[20]-l[0])-\
		w[31]*(w[1]+w[6]+w[11]+w[16]+w[21]-l[1])-\
		w[32]*(w[2]+w[7]+w[12]+w[17]+w[22]-l[2])-\
		w[33]*(w[3]+w[8]+w[13]+w[18]+w[23]-l[3])-\
		w[34]*(w[4]+w[9]+w[14]+w[19]+w[24]-l[4])-\
		miu*sum(np.log(w[:25]))

def dfunc(w,a,l,miu):
	#构造func函数的导数，有两种思路
	#第一种手动的求解导数，这适用于函数的表达式由初等函数组合而成且不很复杂的情形
	#用数值法 dfunc(w)~(func（w+dw)-func(w-dw))/2dw
	#我们采用第一种做法，因为第二种做法有风险，有可能会使（w-dw）出现负数项，从而使得全部结果变为nan	
	dlambda=np.zeros(len(w))
	for i in range (len(w)):
		if(i<=4):
			dlambda[i]=np.log(w[i])+1-w[25]-w[30+i]-miu*1/w[i]
		elif(5<=i and i<10):
			dlambda[i]=np.log(w[i])+1-w[26]-w[25+i]-miu*1/w[i]
		elif(10<=i and i<15):
			dlambda[i]=np.log(w[i])+1-w[27]-w[20+i]-miu*1/w[i]
		elif(15<=i and i<20):
			dlambda[i]=np.log(w[i])+1-w[28]-w[15+i]-miu*1/w[i]
		elif(20<=i and i<25):
			dlambda[i]=np.log(w[i])+1-w[29]-w[10+i]-miu*1/w[i]
		elif(25<=i and i<30 ):
			dlambda[i]=-w[(i-25)*5]-w[(i-25)*5+1]-w[(i-25)+2]-w[(i-25)*5+3]-w[(i-25)*5+4]+a[i-25]
		else:
			dlambda[i]=-w[(i-30)]-w[(i-25)]-w[(i-20)]-w[(i-15)]-w[(i-10)]+l[i-30]
	return dlambda		
#定初值，选取一组满足边界条件的值
import sympy as sp
e=1e-10
A=sp.Matrix([[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			 [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			 [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
			 [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			 [0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
			 [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
			 [0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
			 [0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]])
b=sp.Matrix([[26289],[13463],[732152],[7528],[28382],[0],[207017],[255588],[329515],[15694]])
w=sp.symbols('w')
w=sp.Matrix(sp.symarray(w,35))
t_=sp.solve(A*w-b)
# 设置初值:必须是0到1之间的数，这是因为资产和负债已经归一化。
w=np.ones(35)*0.5
#设置初始误差
error0=func(w,a,l,miu[0])
#设置停止条件
epsilon=1e-5
#迭代次数
cnt=0
#alpha 和 beta是用来求满足Armijo-Goldsteins条件的t_k值
alpha=0.9
beta=0.8
#初始步长
t=1

while True:
	#设置手动进行循环
	input("输入回车键继续...")
	cnt+=1
	#函数值
	function_value=func(w,a,l,miu[0])
	#梯度
	gradient=dfunc(w,a,l,miu[0])
	#归一化的逆向梯度
	neg_uni_gradient=-gradient/np.linalg.norm(gradient)
	#寻找每一次的步长t_k，满足Armijo-Goldsteins条件，我们采用的方法是backtracking
	#x_(k+1)=x_k+t_k*归一化逆向梯度
	
	while True:
		#如果已经满足Armijo-Goldsteins条件以及所有元素为非负则跳出循环
		if(func(w+t*neg_uni_gradient,a,l,miu[0])<func(w,a,l,miu[0])+\
	 alpha*t*np.dot(gradient,neg_uni_gradient) and np.min((w+t*neg_uni_gradient)[:25])>=0):
			break
		else:
			t=beta*t
			

	w=w+t*neg_uni_gradient
		
	#停止的条件
	error1=func(w,a,l,miu[0])
	if abs(error1-error0)<epsilon:
		break
	else:
		error0=error1
		
	print("最新参数为{},熵为{}".format(np.reshape(w,(7,5)),error0))
	
print("迭代终了,极值点是{},函数值为{}".format(w,func(x)))
print(" 迭代次数是{}".format(cnt))
