#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 00:14:31 2020

@author: shengkailin
"""


#---------------------------------libraries----------------------------------#
import numpy as np
import pywt
import matplotlib.pyplot as plt

#----------------------------------program-----------------------------------#
#sampling frequency in HZ
sampling_freq=100
#tine duration in second
duration=20
#time array
time=np.linspace(0,duration,sampling_freq*duration)
x=np.sin(2*np.pi*time)
plt.figure()
plt.plot(time,x)
#construct the scale vector
widths=np.arange(1,31)
coef,freqs = pywt.cwt(data=x, scales=widths,wavelet='morl')
plt.figure()
plt.imshow(coef,extent=[0,duration,1,31],cmap='PRGn',aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
#plt.figure()
#plt.matshow(coef)
plt.colorbar()
plt.show()
#%%

t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
plt.figure()
plt.plot(t,sig)
widths = np.arange(1, 31)
cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
plt.figure()
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
plt.colorbar()
plt.show() # doctest: +SKIP



