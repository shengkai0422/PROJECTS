#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:37:45 2020

@author: shengkailin
"""


#This tutorial is to show the relation between the fourier transform and fourier 
#series


#-----------------------------------LIBRARY-----------------------------------#
import numpy as np
from scipy import signal
from scipy.fftpack import fft,fftshift,fftfreq,ifft,ifftshift
import matplotlib.pyplot as plt
import sys
from textwrap import wrap


#------------------------------------CLASS------------------------------------#
class Data_generator():
	
	sampling_f=None
	sampling_interval=None
	start_time=None
	end_time=None
	wave_type=None
	time=None

	
	def __init__(self,sampling_f=None,start_time=None,end_time=None,wave_type=None):
		self.sampling_f=sampling_f
		
		self.sampling_interval=1/np.array(self.sampling_f)
		self.start_time=start_time
		self.end_time=end_time
		self.time=[]
		self.x=[]
		self.cutoff_f=None
		for i in range(len(self.sampling_f)):
			 self.time.append(np.arange(self.start_time,self.end_time,self.sampling_interval[i]))
		
		self.time=np.array(self.time)
		self.wave_type=wave_type
		print("the data class is constructed!\n")
	
	def function_generator(self):
		if(self.wave_type=="door"):
			#creat a serie of door function with different sampling rate
			for i in range(len(self.time)):
				if(i==0):
					temp=np.array([0,1])
					self.x.append(temp)
				else:	
					temp=np.ones(len(self.time[i]))
					temp[0:(round(len(self.time[i])/4))]=0
					temp[-(round(len(self.time[i])/4)):]=0
					self.x.append(temp)
			self.x=np.array(self.x)	
			
			fig=plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			cmap=plt.cm.get_cmap('jet',10)
			for i in range(len(self.time)):
				ax.plot(self.time[i],np.ones(len(self.time[i]))*self.sampling_f[i],self.x[i],".",c=cmap(i))
				
			ax.set_xlabel("time")
			ax.set_ylabel("Sampling Frequency(Hz)")
			ax.set_zlabel("door function")
			ax.set_title("door functions with different sampling freqeuncy")
			
		elif(self.wave_type=="trangle"):
			self.x=signal.sawtooth(2*np.pi*self.time,width=0.5)
		else:
			print("the function {} can't be generated!\n".format(self.wave_type))
			sys.exit()
			

		
		

class Fourier():
	
	point_num=None
	data=None
	
	
	def __init__(self,data=None,point_num=None):
		self.data=data
		self.point_num=point_num
		self.sampling_F=self.data.sampling_f
		self.X=[]
		self.xf=[]
		self.recovered_signal=[]
		self.recovered_time=[]
		print("the Fourier Transform class is constructed!\n")
		
	def Transform(self):
		
		fig=plt.figure()
		bx = fig.add_subplot(111, projection='3d')
		cmap=plt.cm.get_cmap('jet',10)
		#performing the DFT
		for i in range(len(self.data.time)):
			self.X.append(fft(self.data.x[i],n=self.point_num))
			#Now the fft is in the order of this: from 0Hz to N/2*resolution_f
			#then goes down to -N/2*resolution_f and goes to 0 
			#with the fftshift we center the 0Hz component:the component now start
			#from -N/2*resolution to N/2*resolution
			self.X[i]=fftshift(self.X[i])
			#creat the frequency vector 
			#use fftfreq function to generate the frequency vector
			#n stands for the point number 1/(d*n) stands for the frequency resolution 
			#1/(d*n)=self.sampling_F/n=>d=self.interval
			self.xf.append(fftfreq(n=self.point_num,d=self.data.sampling_interval[i]))
			#now the fftfreq start from 0 to N/2*resolution_f and then goes to
			#-N/2*resolution and goes to 0 now we want to shift this to vector 
			#start from -N/2*resolution to N/2*resolution
			self.xf[i]=fftshift(self.xf[i])
			#here we use DFT to approximate FT so we need to multiply the DFT result by
			#sampling interval
			bx.plot(self.xf[i],self.sampling_F[i]*np.ones(len(self.xf[i])),self.data.sampling_interval[i]*np.abs(self.X[i]),c=cmap(i))
		
		bx.set_xlabel("Frequency")
		bx.set_ylabel("Cutoff frequency")
		bx.set_zlabel("Fourier Trnsoform")
		bx.set_title("FT door functions with different sampling freqeuncy")



	
	def Recovery(self):
				
		fig=plt.figure()
		#fig2=plt.figure()
		cx = fig.add_subplot(111, projection='3d')
		#dx=fig2.add_subplot(111)
		cmap=plt.cm.get_cmap('jet',10)
		for i in range(len(self.data.time)):
		#we recovery the signal from the only the component below cutoff frequency
		#The component of frequency higher than 10Hz will be zeros.
		#we define a door filter window
		#lowpass_filter=np.zeros(len(self.X))
		#find the index of the point where the frequency is between -10 and 10
		#idx_f,=np.where((-cutoff_f<self.xf)&(self.xf<cutoff_f))
		#give lowpass filter value 1 at these points
		#lowpass_filter[idx_f]=1
		#self.lowpass_filter=lowpass_filter
		#doing filter multiplication
		#Xf_filtered=np.multiply(lowpass_filter,self.X)
		
		#before doing the inverse DFT we need to reshift the DFT component to 
			self.recovered_signal.append(ifft(ifftshift(self.X[i])))
			self.recovered_time.append(np.arange(self.data.start_time,len(self.recovered_signal[i])*self.data.sampling_interval[i],self.data.sampling_interval[i]))
		#print(signal.shape)
		#because we have used the frequency resolution of 50/512 ,so the signal of recovery 
		#has the period of 512/50 which under the sampling_interval 0.02=1/50 it should be 512 point
			cx.plot(self.recovered_time[i][0:10*(i+1)],np.ones(len(self.recovered_signal[i]))[0:10*(i+1)]*self.data.sampling_f[i],self.recovered_signal[i][0:10*(i+1)],c=cmap(i))
			
		cx.set_xlabel("time")
		cx.set_ylabel("sampling frequency(Hz)")
		cx.set_zlabel("recovered_signal")
		cx.set_title("recovered signal using different sampling frequency")
	
	#def Recovery_2():
		
	def Gibbs_phenomena(self):
		#we choose to filter the FT that we obtained by the highest sampling frequency 
		#we choose a series of frequency from 0 to the maximum 
		F_max=max(self.data.sampling_f)
		print(F_max)
		#give a vector of 10 cutoff frequency 
		self.cutoff_f=np.arange(0.1*F_max,0.9*F_max,5)
		#each scatter use a different color to better visulize the dataset
		cmap=plt.cm.get_cmap('jet',10)
		fig=plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for i in range(len(self.cutoff_f)):
			#this part is similar to the function Recovery
			lowpass_filter=np.zeros(len(self.X[-1]))
			idx_f,=np.where((-self.cutoff_f[i]<self.xf[-1])&(self.xf[-1]<self.cutoff_f[i]))
			lowpass_filter[idx_f]=1
			Xf_filtered=np.multiply(lowpass_filter,self.X[-1])
			recovered_signal=ifft(ifftshift(Xf_filtered))
			time=np.arange(self.data.start_time,len(recovered_signal)*self.data.sampling_interval[-1],self.data.sampling_interval[-1])
			ax.plot(time,np.ones(len(time))*self.cutoff_f[i],recovered_signal,c=cmap(i))
		
		ax.set_xlabel('time')
		ax.set_ylabel('cutoff frequency(sampling frequency)')
		ax.set_zlabel('recovered signal')
		ax.set_title("Gibbs Phenomena")
		


#-----------------------------------PROGRAM-----------------------------------#	
Sampling_frequency=[1,2,4,8,16,32]
#The fundamental of freqeuncy of the signal is 1Hz,this means that the nyquist 
#frequency should be 2Hz, so the minimum sampling frequency is 2Hz
#From this example we can see that with a high sampling frequency the bandwith is 
#extend very high so as to keep more information about the signals.
data=Data_generator(sampling_f=Sampling_frequency,start_time=0,end_time=2,wave_type="door")
data.function_generator()
model=Fourier(data=data,point_num=128)
model.Transform()	
#compare FFT using different sampling frequency and using different point number
#first we downsample the original signal, this correpond to band filtered the
#FFT of the previous signal
 

#Recovery signal from the DFT
model.Recovery()
#see the Gibbs phenomena
model.Gibbs_phenomena()
#alia














