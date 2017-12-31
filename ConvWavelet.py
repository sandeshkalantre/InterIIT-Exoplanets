# Lines 14 and 15 have the names of the test and train data csv file names. Also uncomment line 37 for the first run. (Preprocesses data)

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

def preprocess():
	# inputting data
	data = np.loadtxt('exoTrain.csv',skiprows=1,delimiter=',')

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
	np.save('preprocessed_wav',[x_train,y_train,dtw_train,x_test,y_test,dtw_test])

# preprocess()

[x_train,y_train,dtw_train,x_test,y_test,dtw_test] = np.load('preprocessed_wav.npy')

dtw_test = np.array(dtw_test).reshape([-1,1])
dtw_train = np.array(dtw_train).reshape([-1,1])

t_train =  []
for i in range(x_train.shape[0]):
	t_train.append(sg.cwt(x_train[i], sg.ricker, np.arange(1,11)))
	print(i)
t_train = np.array(t_train)

t_test = []
for i in range(x_test.shape[0]):
	t_test.append(sg.cwt(x_test[i], sg.ricker, np.arange(1,11)))
	print(i) 

t_test = np.array(t_test)

# # np.save('preprocessed_wav_final',[t_train,y_train,dtw_train,t_test,y_test,dtw_test])
[t_train,y_train,dtw_train,t_test,y_test,dtw_test] = np.load('preprocessed_wav_final.npy')

# @Chinmay: TODO normalisation fir se kar lena
# for i in range(x_test.shape[0]):
# 	t_train[i] = ((t_train[i] - np.mean(t_train[i],axis=1, keepdims = True).reshape(-1,1)) / np.std(t_train[i],axis=1, keepdims = True).reshape(-1,1))
# 	t_test[i] = ((t_test[i] - np.mean(t_test[i],axis=1, keepdims = True).reshape(-1,1)) / np.std(t_test[i],axis=1, keepdims = True).reshape(-1,1))
# np.save('preprocessed_final_augmented.npy',[x_train,y_train,x_test,y_test])
# [x_train,y_train,x_test,y_test] = np.load('preprocessed_final_augmented.npy')

# Parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 100
display_step = 1

# Network Parameters
num_input = np.array([10,x_train.shape[1]])
num_classes = 2
dropout = 0.80 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder("float", [None, num_input[0],num_input[1]])
Y = tf.placeholder("float", [None, num_classes])
dtw = tf.placeholder("float", [None, 1])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

out1 = 48
out2 = 32
out3 = 128
print("yaha tak theek hai")
# Store layers weight & bias
weights = {
	# 1x20 conv, 1 input, 32 outputs
	'wc1': tf.Variable(tf.random_normal([5, 20, 1, out1])),
	'wc2': tf.Variable(tf.random_normal([5, 20, out1, out2])),
	# fully connected,  inputs, 1024 outputs
	'wd1': tf.Variable(tf.random_normal([((((num_input[0]+1)/2)+1)/2)*((((num_input[1]+1)/2)+1)/2)*out2+1, out3])),
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
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='SAME')

# Create model
def conv_net(x,DTW,dropout):
	x = tf.reshape(x, shape=[-1, num_input[0], num_input[1], 1])
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

# Test model
pred = tf.nn.softmax(logits)  # Apply softmax to logits
argmax_prediction = tf.argmax(pred, 1)
argmax_y = tf.argmax(Y, 1)

TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits, labels=tf.argmax(Y,axis = 1)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()


print("starting conv net")
with tf.Session() as sess:
	sess.run(init)
	# tf.train.Saver()

	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch =  102#int(len(x_train)/batch_size)
		X_batches = np.array_split(t_train, total_batch)
		Y_batches = np.array_split(y_train, total_batch)
		dtw_batches = np.array_split(dtw_train,total_batch)

		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y, batch_dtw = X_batches[i], Y_batches[i], dtw_batches[i]
			# Run optimization op (backprop) and cost op (to get loss value)
			print(X_batches[i].shape)
			_, c = sess.run([train_op, loss_op], feed_dict={
															Y: batch_y
															,keep_prob : dropout
															,dtw: batch_dtw
															,X: batch_x
															})
			# Compute average loss
			print(str(i)+" : "+str(c))
			avg_cost += c / total_batch
		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
			# if avg_cost < 100.0:
			print("F1-test:", f1.eval({X: t_test, Y: y_test, keep_prob: 1.0, dtw: dtw_test}))
		# if avg_cost < cost_thresh:
		#     break
	print("Optimization Finished!")

	save_path = tf.train.Saver().save(sess, "/tmp/model_wav.ckpt")
	print("Model saved in file: %s" % save_path)

	prediction = pred.eval({X: t_test, Y: y_test, dtw: dtw_test, keep_prob: 1.0})
	print(np.argmax(prediction,axis = 1))
	print("TP:", TP.eval({X: t_test, Y: y_test, dtw: dtw_test, keep_prob: 1.0}))
	print("TN:", TN.eval({X: t_test, Y: y_test, dtw: dtw_test, keep_prob: 1.0}))
	print("FP:", FP.eval({X: t_test, Y: y_test, dtw: dtw_test, keep_prob: 1.0}))
	print("FN:", FN.eval({X: t_test, Y: y_test, dtw: dtw_test, keep_prob: 1.0}))
	print("precision:", precision.eval({X: t_test, Y: y_test, dtw: dtw_test, keep_prob: 1.0}))
	print("recall:", recall.eval({X: t_test, Y: y_test, dtw: dtw_test, keep_prob: 1.0}))
	print("F1:", f1.eval({X: t_test, Y: y_test, dtw: dtw_test, keep_prob: 1.0}))

	# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	# Calculate accuracy
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	# print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))


