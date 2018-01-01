""" 
Adapted from
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.fftpack as fft
from fastdtw import fastdtw
import scipy.signal as sg
from scipy.spatial.distance import euclidean

[x_train,y_train,dtw_train,x_test,y_test,dtw_test] = np.load('preprocessed_final.npy')
# dtw_test = dtw_test.reshape([-1,1])
# dtw_train = dtw_train.reshape([-1,1])

# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 300
display_step = 1
namechar = '2'
models_params = [50,100,1000]
toRestore = False
modelfile = 'models/model'+namechar+'.ckpt'

# Network Parameters
num_input = x_train.shape[1]
num_classes = 2
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
dtw = tf.placeholder("float", [None, 1])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Store layers weight & bias
def weight(out1,out2,out3,name):
	return {
		# 1x20 conv, 1 input, 32 outputs
		'wc1': tf.Variable(tf.random_normal([1, 20, 1, out1]),name=name),
		'wc2': tf.Variable(tf.random_normal([1, 20, out1, out2]),name=name),
		# fully connected,  inputs, 1024 outputs
		'wd1': tf.Variable(tf.random_normal([((((num_input+1)/2)+1)/2)*out2+1, out3]),name=name),
		# 1024 inputs, 10 outputs (class prediction)
		'out': tf.Variable(tf.random_normal([out3, num_classes]),name=name)
	}

def bias(out1,out2,out3,name):
	return {
		'bc1': tf.Variable(tf.random_normal([out1]),name=name),
		'bc2': tf.Variable(tf.random_normal([out2]),name=name),
		'bd1': tf.Variable(tf.random_normal([out3]),name=name),
		'out': tf.Variable(tf.random_normal([num_classes]),name=name)
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
def conv_net_2c1d(x,DTW,dropout,dims,name):
	[out1,out2,out3] = dims
	weights = weight(out1,out2,out3,name)
	biases = bias(out1,out2,out3,name)
	x = tf.reshape(x, shape=[-1, 1, num_input, 1])
	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1, k=2)
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2, k=2)
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]-1])
	fc1 = tf.concat([fc1,DTW],1)
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	 # Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

# Construct model
logits = conv_net_2c1d(X,dtw,keep_prob,models_params,namechar)

def softmax(X,axis):
	y = np.atleast_2d(X)
	y = y * float(1.0)
	y = y - np.expand_dims(np.max(y, axis = axis), axis)
	y = np.exp(y)
	ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
	p = y / ax_sum
	if len(X.shape) == 1: p = p.flatten()
	return p

def f1(logits_pred,y):
	# print(type(logits_pred))
	pred = softmax(logits_pred,axis=1)
	argmax_prediction = np.argmax(pred,axis=1)
	argmax_y = np.argmax(y,1)
	TP = float(np.count_nonzero(np.rint(argmax_prediction * argmax_y)))
	TN = float(np.count_nonzero(np.rint((argmax_prediction - 1) * (argmax_y - 1))))
	FP = float(np.count_nonzero(np.rint(argmax_prediction * (argmax_y - 1))))
	FN = float(np.count_nonzero(np.rint((argmax_prediction - 1) * argmax_y)))
	if TP + FP == 0 or TP + FN == 0:
		return [None,[TP,TN,FP,FN]]
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	if precision + recall == 0:
		return [None,[TP,TN,FP,FN]]
	f1 = 2 * precision * recall / (precision + recall)
	return [f1,[TP,TN,FP,FN]]

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits, labels=tf.argmax(Y,axis = 1)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	if toRestore == True:
		tf.train.Saver([v for v in tf.global_variables() if v.name[:1]==namechar]).restore(sess, modelfile)
	# Training cycle
	writer = tf.summary.FileWriter('./graph', sess.graph);writer.close();
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(len(x_train)/batch_size)
		X_batches = np.array_split(x_train, total_batch)
		Y_batches = np.array_split(y_train, total_batch)
		dtw_batches = np.array_split(dtw_train,total_batch)

		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y, batch_dtw = X_batches[i], Y_batches[i], dtw_batches[i]
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([train_op, loss_op], feed_dict={
															X: batch_x
															,Y: batch_y
															,keep_prob : dropout
															,dtw: batch_dtw
															})
			# logits_test = logits.eval({X: batch_x, Y: batch_y, keep_prob: 1.0, dtw: batch_dtw})
			# print("F1-test:", f1(logits_test,batch_y))
			# Compute average loss
			avg_cost += c / total_batch
		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
	print("Optimization Finished!")
	logits_test = logits.eval({X: x_test, Y: y_test, keep_prob: 1.0, dtw: dtw_test})
	print("F1-test:", f1(logits_test,y_test))
	save_path = tf.train.Saver([v for v in tf.global_variables() if v.name[:1]==namechar]).save(sess, modelfile)
	print("Model saved in file: %s" % save_path)