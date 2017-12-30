from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.fftpack as fft
from fastdtw import fastdtw
import scipy.signal as sg
from scipy.spatial.distance import euclidean

[x_train,y_train,dtw_train,x_test,y_test,dtw_test] = np.load('preprocessed.npy')
x_test=x_train[:500]
y_test=y_train[:500]
dtw_test = dtw_train[:500]
x_test=np.concatenate((x_test,x_train[:500]))
y_test=np.concatenate((y_test,y_train[:500]))
dtw_test=np.concatenate((dtw_test,dtw_train[:500]))
x_train = x_train[500:]
y_train = y_train[500:]
dtw_train = dtw_train[500:]
dtw_test = np.array(dtw_test).reshape([-1,1])
dtw_train = np.array(dtw_train).reshape([-1,1])
x_train = fft.dct(x_train)
x_train = x_train[:,20:800]
x_test = fft.dct(x_test)
x_test = x_test[:,20:800]
x_train = ((x_train - np.mean(x_train,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_train,axis=1, keepdims = True).reshape(-1,1))
x_test = ((x_test - np.mean(x_test,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_test,axis=1, keepdims = True).reshape(-1,1))

# Parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 100
display_step = 1

# Network Parameters
num_input = x_train.shape[1]
num_classes = 2
dropout = 0.80 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
dtw = tf.placeholder("float", [None, 1])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

out1 = 48
out2 = 32
out3 = 128

# Store layers weight & bias
weights = {
	# 1x20 conv, 1 input, 32 outputs
	'wc1': tf.Variable(tf.random_normal([1, 20, 1, out1])),
	'wc2': tf.Variable(tf.random_normal([1, 20, out1, out2])),
	# fully connected,  inputs, 1024 outputs
	'wd1': tf.Variable(tf.random_normal([((((num_input+1)/2)+1)/2)*out2+1, out3])),
	# 1024 inputs, 10 outputs (class prediction)
	'out': tf.Variable(tf.random_normal([out3, num_classes]))
}

biases = {
	'bc1': tf.Variable(tf.random_normal([out1])),
	'bc2': tf.Variable(tf.random_normal([out2])),
	'bd1': tf.Variable(tf.random_normal([out3])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)


def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1, 1, k, 1],
						  padding='SAME')

# Create model
def conv_net(x,DTW,dropout):
	x = tf.reshape(x, shape=[-1, 1, num_input, 1])
	print(x.shape)
	print(tf.shape(x))
	print("")
	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	print(conv1.shape)
	print(tf.shape(conv1))
	print("")
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1, k=2)
	print(conv1.shape)
	print(tf.shape(conv1))
	print("")
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	print(conv2.shape)
	print(tf.shape(conv2))
	print("")
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2, k=2)
	print(conv2.shape)
	print(tf.shape(conv2))
	print("")
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]-1])
	print(fc1.shape)
	print(tf.shape(fc1))
	print("")
	fc1 = tf.concat([fc1,DTW],1)
	print(fc1.shape)
	print(tf.shape(fc1))
	print("")
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	print(fc1.shape)
	print(tf.shape(fc1))
	print("")
	fc1 = tf.nn.relu(fc1)
	print(fc1.shape)
	print(tf.shape(fc1))
	print("")
	 # Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	print(fc1.shape)
	print(tf.shape(fc1))
	print("")
	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

# Construct model
logits = conv_net(X,dtw,keep_prob)