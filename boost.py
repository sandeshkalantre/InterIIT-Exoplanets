from __future__ import print_function
import tensorflow as tf
import numpy as np

# Load train data here
[x_train,y_train,dtw_train,x_test,y_test,dtw_test] = np.load('preprocessed_final.npy')

x_train = np.concatenate((x_train,x_test))
y_train = np.concatenate((y_train,y_test))
dtw_train = np.concatenate((dtw_train,dtw_test))

n_fold = 6

x_batches_div = np.array_split(x_train, n_fold)
y_batches_div = np.array_split(y_train, n_fold)
dtw_batches_div = np.array_split(dtw_train, n_fold)

for i in range(n_fold):
	x_test = x_batches_div[i]
	y_test = y_batches_div[i]
	dtw_test = dtw_batches_div[i]
	x_train = []
	y_train = []
	dtw_train = []
	for j in range(n_fold):
		if j != i:
			if x_train == []:
				x_train = x_batches_div[j]
				y_train = y_batches_div[j]
				dtw_train = dtw_batches_div[j]
			else:
				x_train = np.concatenate((x_train,x_batches_div[j]))	
				y_train = np.concatenate((y_train,y_batches_div[j]))	
				dtw_train = np.concatenate((dtw_train,dtw_batches_div[j]))	


	# Training Parameters
	learning_rate = 0.00003
	training_epochs = 000
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

	# These are the placeholders for logit outputs of individual models
	d1 = tf.placeholder("float", [None, 2])
	d2 = tf.placeholder("float", [None, 2])
	namechar1 = '4'
	namechar2 = '5'
	# Construct individual models with the respective model parameters
	logits_1 = conv_net_2c1d(X,dtw,keep_prob,50,120,1000,namechar1,False)
	logits_2 = conv_net_2c1d(X,dtw,keep_prob,60,140,1000,namechar2,False)
	# In the first run, it evaluates the corresponding outputs of all the models and saves them in ds.npy
	first_run = False

	# # # If we want single layer neural network
	wd=tf.Variable(tf.random_normal([4,2]))
	bd=tf.Variable(tf.random_normal([2]))
	logits = tf.add(tf.matmul(tf.concat([d1,d2],1), wd),bd)

	# # boosting using weighted average case
	#w1 = tf.Variable(tf.random_normal([1]),name = 'w1')
	#w2 = tf.Variable(tf.random_normal([1]),name = 'w2')
	#logits = (w1*d1+w2*d2)/(w1+w2)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=logits, labels=tf.argmax(Y,axis = 1)))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Initializing the variables
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		if first_run==False:
			tf.train.Saver().restore(sess,"models/model_boost_neural.ckpt")
		tf.train.Saver([v for v in tf.global_variables() if v.name[:1]==namechar1]).restore(sess, "models/model"+namechar1+".ckpt")
		tf.train.Saver([v for v in tf.global_variables() if v.name[:1]==namechar2]).restore(sess, "models/model"+namechar2+".ckpt")
		# writer = tf.summary.FileWriter('./graph', sess.graph);writer.close();
		total_batch = int(len(x_train)/batch_size)
		total_batch1 = int(len(x_test)/batch_size)
		X_batches = np.array_split(x_train, total_batch)
		Y_batches = np.array_split(y_train, total_batch)
		dtw_batches = np.array_split(dtw_train,total_batch)
		X_test_batches = np.array_split(x_test, total_batch1)
		Y_test_batches = np.array_split(y_test, total_batch1)
		dtw_test_batches = np.array_split(dtw_test,total_batch1)
		d1_batches = [None]*total_batch
		d2_batches = [None]*total_batch
		# Even the test data is divided into batches for support of bigger test sample
		X_test_batches = np.array_split(x_test, total_batch1)
		Y_test_batches = np.array_split(y_test, total_batch1)
		dtw_test_batches = np.array_split(dtw_test,total_batch1)
		d1_test_batches = [None]*total_batch1
		d2_test_batches = [None]*total_batch1
		if first_run==True:
	            print("thenga")
			#for i in range(total_batch):
			#	print(i)
			#	[d1_batches[i],d2_batches[i]] = sess.run([logits_1,logits_2],feed_dict={
			#			X: X_batches[i]
			#			,Y: Y_batches[i]
			#			,keep_prob : 1.0
			#			,dtw: dtw_batches[i]
			#		})
			#for i in range(total_batch1):
			#	print(i)
			#	[d1_test_batches[i],d2_test_batches[i]] = sess.run([logits_1,logits_2],feed_dict={
			#			X: X_test_batches[i]
			#			,Y: Y_test_batches[i]
			#			,keep_prob : 1.0
			#			,dtw: dtw_test_batches[i]
			#		})
			#np.save('ds',[d1_batches,d2_batches,d1_test_batches,d2_test_batches])
		
	        # Evaluate the ds's and save for the first time
	        
	        
	        [d1_batches,d2_batches,d1_test_batches,d2_test_batches] = np.load('ds.npy')
		d1_test_batches = np.array(d1_test_batches)
		d2_test_batches = np.array(d2_test_batches)
		for epoch in range(training_epochs):
			avg_cost = 0.
			# Loop over all batches
			for i in range(total_batch):
				d1_val_batch,d2_val_batch = d1_batches[i],d2_batches[i]
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([train_op, loss_op], feed_dict={
																d1: d1_batches[i],
																d2: d2_batches[i],
																Y: Y_batches[i]
															})
				logits_test = logits.eval({
											d1: d1_batches[i],
											d2: d2_batches[i],
											Y: Y_batches[i]
										})
				print("F1-test:", f1(logits_test,Y_batches[i]))
				# Compute average loss
				avg_cost += c / total_batch
			# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
				for i in range(total_batch1):
					logits_test = logits.eval({
										d1: d1_test_batches[i],
										d2: d2_test_batches[i],
										Y: Y_test_batches[i]
									})
					# print(logits_test)
					print("F1-test:", f1(logits_test,Y_test_batches[i]))
		print("Optimization Finished!")
		for i in range(total_batch1):
			logits_test = logits.eval({
								d1: d1_test_batches[i],
								d2: d2_test_batches[i],
								Y: Y_test_batches[i]
							})
			# print(logits_test)
			print("F1-test:", f1(logits_test,Y_test_batches[i]))
		save_path = tf.train.Saver().save(sess, "models/model_boost_neural.ckpt")
		print("Model saved in file: %s" % save_path)
