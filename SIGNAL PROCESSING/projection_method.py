#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:44:36 2020

@author: shengkailin
"""
import numpy as np

#采用Douglas-Rachford splitting 方法
#问题看成是 minf_1(w)+f_2(w),其中f_1是一个指示函数，它指明了不等关系和等式关系，f_2是负熵函数
def negative_entropy(w):
	return sum(w*np.log(w))

def projector_positive(w):
	#小于0的值都投影成0 
	#
	return np.maximum(w, 0)
def projector_equality(w,A,b):
	#把w投影到平面上Aw-b=0
	return w+A.T.dot(np.linalg.pinv(A.dot(A.T))).dot(b-A.dot(w))

def gradient_entropy(w):
	#计算负熵的梯度
	return np.log(w)+np.ones(len(w))
	
w=np.ones(25)*1
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
#更新率
lr=1e-3
#循环次数计数器
cnt=0
while True:
	cnt+=1
	#设置手动进行循环
	input("输入回车键继续...")
	#梯度
	gradient=gradient_entropy(w)
	#归一化的逆向梯度
	neg_uni_gradient=-gradient/np.linalg.norm(gradient)
	#梯度下降
	w=w+lr*neg_uni_gradient
	#投影保证元素非负
	w=projector_positive(w)
	#投影保证等式约束关系
	w=projector_equality(w,A,b)
	#计算熵
	entropy=-negative_entropy(w)

	print("最新参数为{},熵为{}".format(w,entropy))	
print("迭代终了,极值点是{},函数值为{}".format(x,func(x)))
print(" 迭代次数是{}".format(cnt))