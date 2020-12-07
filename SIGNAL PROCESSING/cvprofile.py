#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:09:13 2020

@author: shengkailin
"""

#处理个人简历中的图像
import cv2
import matplotlib.pyplot as plt
import numpy as np
alpha = 0.3
beta = 80
img_path = "/Users/shengkailin/Downloads/touxiang.png"
img = cv2.imread(img_path)
img2 = cv2.imread(img_path)
def updateAlpha(x):
    global alpha, img, img2
    alpha = cv2.getTrackbarPos('Alpha', 'image')
    alpha = alpha * 0.01
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
def updateBeta(x):
    global beta, img, img2
    beta = cv2.getTrackbarPos('Beta', 'image')
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
# 创建窗口
cv2.namedWindow('image')
cv2.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)
cv2.createTrackbar('Beta', 'image', 0, 255, updateBeta)
cv2.setTrackbarPos('Alpha', 'image', 100)
cv2.setTrackbarPos('Beta', 'image', 10)
while (True):
    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()


#I=cv2.imread("/Users/shengkailin/Downloads/touxiang.png")
#转bgr成rgb
#I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
#plt.imshow(I)
