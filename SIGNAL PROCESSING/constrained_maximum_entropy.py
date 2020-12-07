#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:42:12 2020

@author: shengkailin
"""
import numpy as np

#本程序参考了https://zhuanlan.zhihu.com/p/264515249
#该算法使用拉格朗日乘数，对数障碍法以及牛顿优化方法。
def constrained_entropy(w,v,t,A,b):

	"""
	w是双边联系矩阵的列向量,v是拉格朗日函数的乘数行向量，a列向量代表归一化的资产
	t表示障碍函数的系数，它的大小决定着解的精确程度，本例子中当t逐渐变大时，
	数值解无穷趋近于理论上的极小值
	l是列向量归一化的负债
	numpy 里的列向量是形如“array([1,2,3])”的numpy向量，矩阵A与列v相乘使用A.dot(v)
	"""
	#约束有两部分，一部分是障碍对数函数，它用来限制双边联系矩阵的函数为非负，
	#另一部分是拉格朗日乘数，它来规定矩阵行或列的元素和为归一化资产或者负债
	#对数障碍
	constrained_barrier=-1/t*sum(np.log(w))
	#拉格朗日函数中的等式约束部分,形式上是:v(Aw-b)，其中v是行向量，
	#A是矩阵（25*10:25个矩阵变量，10个约束方程)，b是列向量
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
	b=np.concatenate((a,l),axis=0)		
	constrained_lagrange=(A.dot(w)-b).dot(v)
	#普通熵的定义
	entropy=-sum(w*np.log(w))
	#约束的熵由三部分组成：一是普通熵的相反数（这是因为我们是要找熵的最大值，等价于找负熵的最小值）
	#二是对数障碍（它的目的是限制所有矩阵元素为非负），三是拉格朗日函数中的等式约束部分
	constrained_entropy=-entropy+constrained_barrier+constrained_lagrange
	return constrained_entropy

def gradient_lagrangian(w,v,t,A,b):
	
	"""
	该函数为了求出拉格朗日函数的梯度向量
	其中w是双边联系矩阵中元素组成的向量，A,b是等式约束矩阵方程里的系数 Aw-b=0,它规定着双边联系矩阵中行或列
	的和等于归一化资产或者负债
	"""
	#首先初始化一个空向量，用来承载各个分量上梯度计算值
	size=len(w)+len(v)
	gradient=np.zeros(size)
	#这里由于拉格朗日函数是由初等函数组成的所以我们直接笔算了它的梯度
	#我们把梯度的计算分成两部分，第一部分是涉及到对于w元素的导数，第二部分涉及到对于拉格朗日乘数v
	#的导数
	gradient_w=1+np.log(w)+1/(t*w)-A.T.dot(v)
	gradient_v=A.dot(w)-b
	#然后合并向量
	gradient=np.concatenate((gradient_w,gradient_v),axis=0)
	return gradient
def hessian_lagrangian(w,v,t,A,b):
	"""
	该函数为了求出拉格朗日函数的hessian矩阵，该矩阵为拉格朗日函数对于各变量的二阶导数，这个矩阵将
	在今后的牛顿优化法中使用
	其中w是双边联系矩阵中元素组成的向量，A,b是等式约束矩阵方程里的系数 Aw-b=0,它规定着双边联系矩阵中行或列
	的和等于归一化资产或者负债	

	"""
	#我们还是采用求部分矩阵然后合成的方法，首先考虑拉格朗日函数对于诸w的二阶导数，一个有意思的特点是
	#由于拉格朗日函数内每一个w_i仅仅与自己发生相乘（或者先做对数运算再与自己相乘），因而所有的二阶混合导数均为0，
	#仅有连续对一个固定量w_i求的二阶导数不为0
	hessian_w=np.diag(w**(-1))+np.diag(w**(-2))
	#与A的转秩，A以及0矩阵的合并最终得到拉格朗日函数的hessian矩阵
	temp=np.concatenate((hessian_w,A.T),axis=1)
	size=len(v)
	temp_=np.concatenate((A,np.zeros((size,size))),axis=1)
	hessian=np.concatenate((temp,temp_),axis=0)
	return hessian
def newton_optimization(gradient,hessian,t,w,v,A,b):
	"""
牛顿法是在某点的附近通过用二次曲面来寻找局部最小值的方法不断逼近全局最小值

	"""
	#先初始化两个空矩阵用来承接二次曲面局部最小值
	delta_w=np.zeros(len(w))
	delta_v=np.zeros(len(v))
	temp=-np.linalg.inv(hessian).dot(gradient)
	delta_w=temp[:len(w)]
	delta_v=temp[len(w):]
	return delta_w,delta_v

#---------------
t=1000
miu=2
w=np.ones(25)*0.5
v=np.ones(10)*0.5
#等式约束方程的个数
m=10
#终止条件
epsilon=1e-4
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
cnt=0
while True:
	cnt+=1
	#设置手动进行循环
	input("输入回车键继续...")
	entropy=constrained_entropy(w,v,t,A,b)
	gradient=gradient_lagrangian(w,v,t,A,b)
	hessian=hessian_lagrangian(w,v,t,A,b)
	delta_w,delta_v=newton_optimization(gradient,hessian,t,w,v,A,b)
	
	w=w-delta_w
	v=v-delta_v
	if(m/t<=epsilon):
		break
	else:
		t=miu*t
	print("最新参数为{},熵为{}".format(w,entropy))	
print("迭代终了,极值点是{},函数值为{}".format(x,func(x)))
print(" 迭代次数是{}".format(cnt))
	