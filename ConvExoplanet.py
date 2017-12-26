# Lines 14 and 15 have the names of the test and train data csv file names. Also uncomment line 37 for the first run. (Preprocesses data)

""" 
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.fftpack as fft

def preprocess():
    # inputting data
    data = np.loadtxt('ExoTrain.csv',skiprows=1,delimiter=',')
    data = np.random.shuffle(data)
    x_train = data[:,1:]
    y_train = data[:, 0, np.newaxis] - 1
    y_train = np.concatenate((1-y_train,y_train),axis=1).astype(int)
    data = np.loadtxt('exoTest.csv',skiprows=1,delimiter=',')
    x_test = data[:,1:]
    y_test = data[:, 0, np.newaxis] - 1
    y_test = np.concatenate((1-y_test,y_test),axis=1).astype(int)
    del data
    print('Data inputted')
    print('Preprocessing data')
    import scipy.signal as sg
    for  i in range(x_train.shape[0]):
        if i%100 == 0 :
            print(i)
        x_train[i] = x_train[i] - sg.medfilt(x_train[i],101)
    for  i in range(x_test.shape[0]):
        if i%100 == 0 :
            print(i)
        x_test[i] = x_test[i] - sg.medfilt(x_test[i],101)
    print('Preprocessing done')
    np.save('preprocessed',[x_train,y_train,x_test,y_test])
# preprocess()
[x_train,y_train,x_test,y_test] = np.load('preprocessed.npy')

x_train = x_train[:,101:3100]
x_test = x_test[:,101:3100]

x_train = fft.dct(x_train)
# x_train[:,:20] = 0
# x_train[:,800:] = 0
# x_train = fft.idct(x_train)
x_train = x_train[:,20:800]
x_test = fft.dct(x_test)
# x_test[:,:20] = 0
# x_test[:,800:] = 0
# x_test = fft.idct(x_test)
x_test = x_test[:,20:800]

# # trying without the dct thing
x_train = ((x_train - np.mean(x_train,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_train,axis=1, keepdims = True).reshape(-1,1))
x_test = ((x_test - np.mean(x_test,axis=1, keepdims = True).reshape(-1,1)) / np.std(x_test,axis=1, keepdims = True).reshape(-1,1))

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

# Network Parameters
num_input = x_train.shape[1]
num_classes = 2
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Store layers weight & bias
weights = {
    # 1x20 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([1, 20, 1, 32])),
    # 1x20 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([1, 20, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([((((num_input+1)/2)+1)/2)*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
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
def conv_net(x,dropout):
    x = tf.reshape(x, shape=[-1, 1, num_input, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
     # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Construct model
logits = conv_net(X,keep_prob)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(x_train)/batch_size)
        X_batches = np.array_split(x_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)


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

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y,
                                                            keep_prob : dropout})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            print("F1:", f1.eval({X: x_test, Y: y_test,keep_prob: 1.0}))
        # if avg_cost < cost_thresh:
        #     break
    print("Optimization Finished!")

    prediction = pred.eval({X: x_test, Y: y_test, keep_prob: 1.0})
    print(np.argmax(prediction,axis = 1))
    print("TP:", TP.eval({X: x_test, Y: y_test, keep_prob: 1.0}))
    print("TN:", TN.eval({X: x_test, Y: y_test, keep_prob: 1.0}))
    print("FP:", FP.eval({X: x_test, Y: y_test, keep_prob: 1.0}))
    print("FN:", FN.eval({X: x_test, Y: y_test, keep_prob: 1.0}))
    print("precision:", precision.eval({X: x_test, Y: y_test,keep_prob: 1.0}))
    print("recall:", recall.eval({X: x_test, Y: y_test,keep_prob: 1.0}))
    print("F1:", f1.eval({X: x_test, Y: y_test,keep_prob: 1.0}))

    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
