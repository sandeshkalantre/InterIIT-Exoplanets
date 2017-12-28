from __future__ import print_function
#import tensorflow as tf
import numpy as np
import scipy.fftpack as fft
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy import signal 


data = np.loadtxt('exoTrain.csv', skiprows=1, delimiter=',')
np.random.shuffle(data)
x_train = data[:,1:]
y_train = data[:,0,np.newaxis] -1 
y_train = np.concatenate((1-y_train,y_train),axis=1).astype(int)
data = np.loadtxt('exoTest.csv',skiprows=1,delimiter=',')
x_test = data[:,1:]
y_test = data[:, 0, np.newaxis] - 1
y_test = np.concatenate((1-y_test,y_test),axis=1).astype(int)
del data

# x_train = ((x_train - np.mean(x_train,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_train,axis=1, keepdims = True).reshape(-1,1))
# x_test = ((x_test - np.mean(x_test,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_test,axis=1, keepdims = True).reshape(-1,1))




def dtw_add(data):
	
	avg=[]
	std=[]
	dist_temp = []
	path_temp = []

	template = np.linspace(1, data.shape[1], 1, endpoint=False)

	for i in range(data.shape[0]):
		avg.append(np.mean(data[i]))
		std.append(np.std(data[i])) 
			
	
	for i in range(data.shape[0]):
		
		y = -3*std[i]*signal.square(template,duty=0.1)
		y[0] = data[i][0]
		y[-1] = data[i][-1]	
		distance, path = fastdtw(data[i], y, dist=euclidean)
		dist_temp.append(distance)
		path_temp.append(path)
        dist_temp = np.array(dist_temp)
        dist_temp = dist_temp.T
	np.save('dtw_values',[dist_temp])

print(x_train.shape)

dtw_add(x_train)

[dist_temp] = np.load('dtw_values.npy') 
print(x_train.shape)
print(dist_temp.shape)
