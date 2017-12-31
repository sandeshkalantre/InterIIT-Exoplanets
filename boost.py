""" 
Adapted from
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

[x_train,y_train,dtw_train,x_test,y_test,dtw_test] = np.load('preprocessed_final.npy')
# [x_train,y_train,dtw_train] = np.load('preprocessed1.npy')

# Parameters
learning_rate = 0.00003
training_epochs = 1000
batch_size = 300
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

# Store layers weight & bias
def weight(out1,out2,out3,name,isTrainable):
	return {
		# 1x20 conv, 1 input, 32 outputs
		'wc1': tf.Variable(tf.random_normal([1, 20, 1, out1]),name=name,trainable=isTrainable),
		'wc2': tf.Variable(tf.random_normal([1, 20, out1, out2]),name=name,trainable=isTrainable),
		# fully connected,  inputs, 1024 outputs
		'wd1': tf.Variable(tf.random_normal([((((num_input+1)/2)+1)/2)*out2+1, out3]),name=name,trainable=isTrainable),
		# 1024 inputs, 10 outputs (class prediction)
		'out': tf.Variable(tf.random_normal([out3, num_classes]),name=name,trainable=isTrainable)
	}

def bias(out1,out2,out3,name,isTrainable):
	return {
		'bc1': tf.Variable(tf.random_normal([out1]),name=name,trainable=isTrainable),
		'bc2': tf.Variable(tf.random_normal([out2]),name=name,trainable=isTrainable),
		'bd1': tf.Variable(tf.random_normal([out3]),name=name,trainable=isTrainable),
		'out': tf.Variable(tf.random_normal([num_classes]),name=name,trainable=isTrainable)
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
def conv_net_2c1d(x,DTW,dropout,out1,out2,out3,name,isTrainable):
	weights = weight(out1,out2,out3,name,isTrainable)
	biases = bias(out1,out2,out3,name,isTrainable)
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

d1 = tf.placeholder("float", [None, 2])
d2 = tf.placeholder("float", [None, 2])
d3 = tf.placeholder("float", [None, 2])
d4 = tf.placeholder("float", [None, 2])
# Construct model
logits_3 = conv_net_2c1d(X,dtw,keep_prob,40,80,500,'1',False)
logits_4 = conv_net_2c1d(X,dtw,keep_prob,32,16,128,'2',False)
logits_5 = conv_net_2c1d(X,dtw,keep_prob,30,15,100,'3',False)
logits_6 = conv_net_2c1d(X,dtw,keep_prob,48,32,128,'4',False)

# wd=tf.Variable(tf.random_normal([8,2]))
# bd=tf.Variable(tf.random_normal([2]))
w1 = tf.Variable(tf.random_normal([1]),name = 'w1')
w2 = tf.Variable(tf.random_normal([1]),name = 'w2')
w3 = tf.Variable(tf.random_normal([1]),name = 'w3')
w4 = tf.Variable(tf.random_normal([1]),name = 'w4')
# logits_concat = tf.concat([d1,d2,d3,d4],1)
# logits = tf.add(tf.matmul(logits_concat, wd),bd)
logits = (w1*d1+w2*d2+w3*d3+w4*d4)/(w1+w2+w3+w4)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits, labels=tf.argmax(Y,axis = 1)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	tf.train.Saver().restore(sess,"models/model_boost.ckpt")
	# tf.train.Saver([v for v in tf.global_variables() if v.name[:1]=='1']).restore(sess, "models/model3.ckpt")
	# tf.train.Saver([v for v in tf.global_variables() if v.name[:1]=='2']).restore(sess, "models/model4.ckpt")
	# tf.train.Saver([v for v in tf.global_variables() if v.name[:1]=='3']).restore(sess, "models/model5.ckpt")
	# tf.train.Saver([v for v in tf.global_variables() if v.name[:1]=='4']).restore(sess, "models/model6.ckpt")
	# Training cycle
	# writer = tf.summary.FileWriter('./graph', sess.graph);writer.close();
	total_batch = int(len(x_train)/batch_size)
	X_batches = np.array_split(x_train, total_batch)
	Y_batches = np.array_split(y_train, total_batch)
	dtw_batches = np.array_split(dtw_train,total_batch)
	d1_batches = [None]*total_batch
	d2_batches = [None]*total_batch
	d3_batches = [None]*total_batch
	d4_batches = [None]*total_batch
	# for i in range(total_batch):
	# 	print(i)
	# 	[d1_batches[i],d2_batches[i],d3_batches[i],d4_batches[i]] = sess.run([logits_3,logits_4,logits_5,logits_6],feed_dict={
	# 			X: X_batches[i]
	# 			,Y: Y_batches[i]
	# 			,keep_prob : 1.0
	# 			,dtw: dtw_batches[i]
	# 		})
	# [d1_test,d2_test,d3_test,d4_test] = sess.run([logits_3,logits_4,logits_5,logits_6],feed_dict={
	# 		X: x_test
	# 		,Y:y_test
	# 		,keep_prob : 1.0
	# 		,dtw: dtw_test
	# 	})
	# np.save('ds',[d1_batches,d2_batches,d3_batches,d4_batches,d1_test,d2_test,d3_test,d4_test])
	[d1_batches,d2_batches,d3_batches,d4_batches,d1_test,d2_test,d3_test,d4_test] = np.load('ds.npy')
	for epoch in range(training_epochs):
		avg_cost = 0.
		# Loop over all batches
		for i in range(total_batch):
			d1_val_batch,d2_val_batch,d3_val_batch,d4_val_batch = d1_batches[i],d2_batches[i],d3_batches[i],d4_batches[i]
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([train_op, loss_op], feed_dict={
															d1: d1_batches[i],
															d2: d2_batches[i],
															d3: d3_batches[i],
															d4: d4_batches[i],
															Y: Y_batches[i]
														})
			# logits_test = logits.eval({
			# 							d1: d1_batches[i],
			# 							d2: d2_batches[i],
			# 							d3: d3_batches[i],
			# 							d4: d4_batches[i],
			# 							Y: Y_batches[i]
			# 						})
			# print("F1-test:", f1(logits_test,Y_batches[i]))
			# Compute average loss
			avg_cost += c / total_batch
		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
			logits_test = logits.eval({
								d1: d1_test,
								d2: d2_test,
								d3: d3_test,
								d4: d4_test,
								Y: y_test
							})
			print("logits")
			print("F1-test:", f1(logits_test,y_test))
	print("Optimization Finished!")
	save_path = tf.train.Saver().save(sess, "models/model_boost.ckpt")
	print("Model saved in file: %s" % save_path)