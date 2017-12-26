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
learning_rate = 0.0005
training_epochs = 2000
batch_size = 100
display_step = 10

# Network Parameters
n_hidden_1 = 128 # 1st layer number of neurons
n_hidden_2 = 16 # 2nd layer number of neurons
n_hidden_3 = 8 # 2nd layer number of neurons
n_input = x_train.shape[1]
n_classes = 2

cost_thresh = 0.001

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

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
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        # if avg_cost < cost_thresh:
        #     break
    print("Optimization Finished!")

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
    prediction = pred.eval({X: x_test, Y: y_test})
    print(np.argmax(prediction,axis = 1))
    print("TP:", TP.eval({X: x_test, Y: y_test}))
    print("TN:", TN.eval({X: x_test, Y: y_test}))
    print("FP:", FP.eval({X: x_test, Y: y_test}))
    print("FN:", FN.eval({X: x_test, Y: y_test}))
    print("precision:", precision.eval({X: x_test, Y: y_test}))
    print("recall:", recall.eval({X: x_test, Y: y_test}))
    print("F1:", f1.eval({X: x_test, Y: y_test}))

    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
