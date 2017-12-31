'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.fftpack as fft
from fastdtw import fastdtw
import scipy.signal as sg
from scipy.spatial.distance import euclidean
from scipy import signal 


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
        #y = template*np.mean(x_train[i])
        y = 3*np.std(x_train[i])*signal.square(template,duty=0.1)
        y[0] = x_train[i][0]
        y[-1] = x_train[i][-1]  
        distance, path = fastdtw(x_train[i], y, dist=euclidean)
        dtw_train[i] = distance
    
    for  i in range(x_test.shape[0]):
        if i%100 == 0 :
            print(i)
        x_test[i] = sg.medfilt(x_test[i],3)
        x_test[i] = x_test[i] - sg.medfilt(x_test[i],101)
        #y = template*np.mean(x_test[i])
        
        y = 3*np.std(x_train[i])*signal.square(template,duty=0.1)
        y[0] = x_test[i][0]
        y[-1] = x_test[i][-1]   
        distance, path = fastdtw(x_test[i], y, dist=euclidean)
        dtw_test[i] = distance

    print('Preprocessing done')
    np.save('preprocessed',[x_train,y_train,dtw_train,x_test,y_test,dtw_test])

# preprocess()

[x_train,y_train,dtw_train,x_test,y_test,dtw_test] = np.load('preprocessed.npy')

dtw_test = np.log(dtw_test)
dtw_train = np.log(dtw_train)

#x_test=np.concatenate((x_test,x_train[:500]))
y_test=np.concatenate((y_test,y_train[:500]))
dtw_test=np.concatenate((dtw_test,dtw_train[:500]))
#x_train = x_train[500:]

dtw_train = np.array(dtw_train)
y_train = np.array(y_train)
dtw_train_pos = dtw_train[y_train[:,1] > 0]
y_train_pos = y_train[y_train[:,1]>0]
dtw_train_neg = dtw_train[y_train[:,1]==0]
y_train_neg = y_train[y_train[:,1]==0]
y_train = np.concatenate((y_train_pos,y_train_neg[:33]))
dtw_train = np.concatenate((dtw_train_pos,dtw_train_neg[:33]))

dtw_test = np.array(dtw_test).reshape([-1,1])
dtw_train = np.array(dtw_train).reshape([-1,1])

dtw_train = ((dtw_train - np.mean(dtw_train,axis=1, keepdims = True).reshape(-1,1)) / np.std(dtw_train,axis=1, keepdims = True).reshape(-1,1))
dtw_test = ((dtw_test - np.mean(dtw_test,axis=1, keepdims = True).reshape(-1,1)) / np.std(dtw_test,axis=1, keepdims = True).reshape(-1,1))

print(np.std(dtw_train,axis=1, keepdims = True).reshape(-1,1))

# Parameters
learning_rate = 0.1
training_epochs = 50
batch_size = 10
display_step = 1

# Network Parameters
num_input = dtw_train.shape[0]
num_classes = 2

# tf Graph Input
initialize = tf.contrib.layers.xavier_initializer()
X = tf.placeholder(tf.float32, [None, 1]) # mnist data image of shape 28*28=784
Y = tf.placeholder(tf.float32, [None, num_classes]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(initialize([2]))
b = tf.Variable(initialize([2]))

# Construct model
pred = tf.nn.sigmoid(tf.matmul(X, W) + b) # Softmax

argmax_prediction = tf.argmax(pred, 1)
argmax_y = tf.argmax(Y, 1)

TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# Minimize error using cross entropy
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices=1))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.argmax(Y,axis = 1)))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(dtw_train)/batch_size) 
        X_batches  = np.array_split(dtw_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs,
                                                          Y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch 
        # Display logs per epoch step
        if (epoch) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            print("F1-test:", f1.eval(session=sess,feed_dict={X: dtw_test, Y: y_test}))
            # print(W.eval())
            # print(b.eval())  

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval(session=sess,feed_dict={X: dtw_test, Y: y_test}))