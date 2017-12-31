from __future__ import print_function
import numpy as np
import scipy.fftpack as fft
from fastdtw import fastdtw
import scipy.signal as sg
from scipy.spatial.distance import euclidean

def preprocess():
	# inputting data
	data = np.loadtxt('exoTrain.csv',skiprows=1,delimiter=',')
	np.random.shuffle(data)
	x_train = data[:,1:]
	y_train = data[:, 0, np.newaxis] - 1
	y_train = np.concatenate((1-y_train,y_train),axis=1).astype(int)
	dtw_train = [None]*x_train.shape[0]
	data = np.loadtxt('final_Test.csv',skiprows=1,delimiter=',')
	x_test = data[:,1:]
	y_test = data[:, 0, np.newaxis] - 1
	y_test = np.concatenate((1-y_test,y_test),axis=1).astype(int)
	dtw_test = [None]*x_test.shape[0]
	del data
	template = np.linspace(1, x_train.shape[1], 1, endpoint=False)
	print('Data inputted')
	print('Preprocessing data')
	for  i in range(x_train.shape[0]):
		if i%100 == 0 :
			print(i)
		x_train[i] = sg.medfilt(x_train[i],3)
		x_train[i] = x_train[i] - sg.medfilt(x_train[i],101)
		y = template*np.mean(x_train[i])
		y[0] = x_train[i][0]
		y[-1] = x_train[i][-1]	
		distance, path = fastdtw(x_train[i], y, dist=euclidean)
		dtw_train[i] = distance
	for  i in range(x_test.shape[0]):
		if i%100 == 0 :
			print(i)
		x_test[i] = sg.medfilt(x_test[i],3)
		x_test[i] = x_test[i] - sg.medfilt(x_test[i],101)
		y = template*np.mean(x_test[i])
		y[0] = x_test[i][0]
		y[-1] = x_test[i][-1]	
		distance, path = fastdtw(x_test[i], y, dist=euclidean)
		dtw_test[i] = distance
	print('Preprocessing done')
	np.save('preprocessed_iitm',[x_train,y_train,dtw_train,x_test,y_test,dtw_test])

preprocess()
print('reducing')
[temporary,x_train,y_train,dtw_train] = np.load('preprocessed_iitm.npy')
x_train = x_train[:,96:3095]
# x_test = x_test[:,96:3095]

# x_test=x_train[:500]
# y_test=y_train[:500]
# dtw_test = dtw_train[:500]
# # x_test=np.concatenate((x_test,x_train[:500]))
# # y_test=np.concatenate((y_test,y_train[:500]))
# # dtw_test=np.concatenate((dtw_test,dtw_train[:500]))
# x_train = x_train[500:]
# y_train = y_train[500:]
# dtw_train = dtw_train[500:]


# dtw_test = np.array(dtw_test).reshape([-1,1])
dtw_train = np.array(dtw_train).reshape([-1,1])

x_train = fft.dct(x_train)
x_train = x_train[:,20:800]
# x_test = fft.dct(x_test)
# x_test = x_test[:,20:800]

x_train = ((x_train - np.mean(x_train,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_train,axis=1, keepdims = True).reshape(-1,1))
# x_test = ((x_test - np.mean(x_test,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_test,axis=1, keepdims = True).reshape(-1,1))
dtw_train = (dtw_train-np.mean(dtw_train))/np.std(dtw_train)
# dtw_test = (dtw_test-np.mean(dtw_test))/np.std(dtw_test)

np.save('preprocessed_iitm_final',[x_train,y_train,dtw_train])