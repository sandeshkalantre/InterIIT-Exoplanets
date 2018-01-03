"""
https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration
We took reduce_upper_outliers function from the link and it also inspired our preprocessing pipeline
"""
from __future__ import print_function
import numpy as np
import scipy.fftpack as fft
from fastdtw import fastdtw
import scipy.signal as sg
from scipy.spatial.distance import euclidean
import pandas as pd
from scipy.signal import savgol_filter
import scipy

def data_augment(input_dataset,output,slice=5):
	print("running data augmentor")
	output = np.empty([slice*input_dataset.shape[0],(input_dataset.shape[1]-slice +1)])
	for i in range(input_dataset.shape[0]):
		for j in range(1,slice+1):
			output[i*5+j-1][1:] = input_dataset[i][j:(input_dataset.shape[1]-slice+j)]
			output[i*5+j-1][0] = input_dataset[i][0]

	return output 		 



def reduce_upper_outliers(df,reduce = 0.01, half_width=4):
	length = len(df.iloc[0,:])
	remove = int(length*reduce)
	for i in df.index.values:
		values = df.loc[i,:]
		sorted_values = values.sort_values(ascending = False)
	   # print(sorted_values[:30])
		for j in range(remove):
			idx = sorted_values.index[j]
			#print(idx)
			new_val = 0
			count = 0
			idx_num = idx
			# idx_num = int(idx[5:])
			#print(idx,idx_num)
			for k in range(2*half_width+1):
				idx2 = idx_num + k - half_width
				if idx2 <1 or idx2 >= length or idx_num == idx2:
					continue
				new_val += values[idx2] # corrected from 'FLUX-' to 'FLUX.'

				count += 1
			new_val /= count # count will always be positive here
			#print(new_val)
			if new_val < values[idx]: # just in case there's a few persistently high adjacent values
				df.set_value(i,idx,new_val)


	return df

def preprocess():
	# inputting data
	data = np.loadtxt('/net/voxel01/misc/extra/code/jimmy/Exoplanets/Final_Test.csv',skiprows=1,delimiter=',')
	# data = np.loadtxt('exoTrain.csv',skiprows=1,delimiter=',')
	# output = []
	# data = data_augment(data,output,5)
	# print(data.shape)
	np.random.shuffle(data)
	x_train = data[:,1:]
	y_train = data[:, 0, np.newaxis] - 1
	y_train = np.concatenate((1-y_train,y_train),axis=1).astype(int)
	dtw_train = [None]*x_train.shape[0]
	data = np.loadtxt('exoTest.csv',skiprows=1,delimiter=',')
	x_test = data[:,1:]
	y_test = data[:, 0, np.newaxis] - 1
	y_test = np.concatenate((1-y_test,y_test),axis=1).astype(int)
	dtw_test = [None]*x_test.shape[0]
	del data
	x_test = reduce_upper_outliers(pd.DataFrame(x_test)).values
	x_train = reduce_upper_outliers(pd.DataFrame(x_train)).values
	x_test = reduce_upper_outliers(pd.DataFrame(x_test)).values
	x_train = reduce_upper_outliers(pd.DataFrame(x_train)).values
	template = np.linspace(1, x_train.shape[1], 1, endpoint=False)
	print('Data inputted')
	print('Preprocessing data')
	for  i in range(x_train.shape[0]):
		if i%100 == 0 :
			print(i)
		x_train[i] = sg.medfilt(x_train[i],3)
		x_train[i] = x_train[i] - sg.medfilt(x_train[i],101)
		savgol_filter(x_train[i],15,4,deriv=0)
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
		savgol_filter(x_test[i],15,4,deriv=0)
		y = template*np.mean(x_test[i])
		y[0] = x_test[i][0]
		y[-1] = x_test[i][-1]	
		distance, path = fastdtw(x_test[i], y, dist=euclidean)
		dtw_test[i] = distance
	x_train = ((x_train - np.mean(x_train,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_train,axis=1, keepdims = True).reshape(-1,1))
	x_test = ((x_test - np.mean(x_test,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_test,axis=1, keepdims = True).reshape(-1,1))
	dtw_train = (dtw_train-np.mean(dtw_train))/np.std(dtw_train)
	dtw_test = (dtw_test-np.mean(dtw_test))/np.std(dtw_test)
	print('Preprocessing done')
	np.save('preprocessed_iitmdata',[x_train,y_train,dtw_train,x_test,y_test,dtw_test])

preprocess()
print('reducing')
[x_train,y_train,dtw_train,x_test,y_test,dtw_test] = np.load('preprocessed_iitmdata.npy')

x_train = x_train[:,100:3100]
x_test = x_test[:,100:3100]

x_test=x_train[:500]
y_test=y_train[:500]
dtw_test = dtw_train[:500]
x_train = x_train[500:]
y_train = y_train[500:]
dtw_train = dtw_train[500:]

# x_train = np.concatenate((x_train[:,96:3095],x_train[:,91:3090],x_train[:,101:3100]))
# dtw_train = np.concatenate((dtw_train,dtw_train,dtw_train))
# y_train = np.concatenate((y_train,y_train,y_train))
# x_test = x_test[:,96:3095]

dtw_test = np.array(dtw_test).reshape([-1,1])
dtw_train = np.array(dtw_train).reshape([-1,1])

def spectrum_getter(X):
    Spectrum = scipy.fft(X, n=X.size)
    return np.abs(Spectrum)

x_train_temp = [None]*len(x_train)
x_test_temp = [None]*len(x_test)
for i in range(len(x_train)):
	x_train_temp[i] = (spectrum_getter(x_train[i])[0:1500])
	print(i)

for i in range(len(x_test)):
	x_test_temp[i] = (spectrum_getter(x_test[i])[0:1500])
	print(i)

# x_train = fft.dct(x_train)
# x_train = x_train[:,20:800]
# x_test = fft.dct(x_test)
# x_test = x_test[:,20:800]

x_train,x_test = x_train_temp,x_test_temp

x_train = ((x_train - np.mean(x_train,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_train,axis=1, keepdims = True).reshape(-1,1))
x_test = ((x_test - np.mean(x_test,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_test,axis=1, keepdims = True).reshape(-1,1))
dtw_train = (dtw_train-np.mean(dtw_train))/np.std(dtw_train)
dtw_test = (dtw_test-np.mean(dtw_test))/np.std(dtw_test)

np.save('preprocessed_final_iitmdata',[x_train,y_train,dtw_train,x_test,y_test,dtw_test])