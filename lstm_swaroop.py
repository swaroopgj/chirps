'''
A Bidirectional Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import cPickle as pickle
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
with open('song_data.pkl', 'r') as f:
    data = pickle.load(f)

samples = data['data']
classes = [data['labels'][k] for k in data['names']]
classes = np.asarray([[1, 0] if l == 0 else [0, 1] for l in classes])
# remove samples <205 and >250
shapes = np.array([s.shape[1]/128 for s in samples])
filter = (shapes<250) & (shapes>205)
samples = [samples[i] for i in range(len(samples)) if filter[i]]
classes = classes[filter]

# permutate
ord = np.random.permutation(len(samples))
samples = [samples[i] for i in ord]
classes = classes[ord]

del data
'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.0005
training_iters = 500000 #14000 #100000
batch_size = 128
display_step = 100
beta = 0.0001
# Network Parameters
n_input = 128 # MNIST data input (img shape: 28*28)
n_steps = 200 # timesteps
n_hidden = 200 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)
n_layers = 3

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x, weights, biases, keep_prob):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)
    #x = tf.split(1, n_input, x)


    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # Dropout
    lstm_fw_cell = rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
    lstm_bw_cell = rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)
    # stacking cells into layers
    stacked_lstm_fw_cells = rnn_cell.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
    stacked_lstm_bw_cells = rnn_cell.MultiRNNCell([lstm_bw_cell] * n_layers,
                                                  state_is_tuple=True)
    state_fw = stacked_lstm_fw_cells.zero_state(batch_size, tf.float32)
    state_bw = stacked_lstm_bw_cells.zero_state(batch_size, tf.float32)
    # Get lstm cell output
    try:
        outputs, state_fw, state_bw = rnn.bidirectional_rnn(stacked_lstm_fw_cells, stacked_lstm_bw_cells, x,
                                              state_fw, state_bw, statedtype=tf.float32)
        #outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
        #                                      statedtype=tf.float32)

    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    #print(tf.concat(1, [outputs[-1][0], outputs[-1][1]]))
    out_vals = tf.concat(1, [outputs[-1][0], outputs[-1][1]])
    #return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.matmul(out_vals, weights['out']) + biases['out']


pred = BiRNN(x, weights, biases, keep_prob=0.5)

# Define loss and optimizer
cost_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost_l2 = tf.nn.l2_loss(weights['out']) #+ tf.nn.l2_loss(weights['hid'])
cost = cost_softmax + beta*cost_l2
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
training_samples = zip(range(0, 14000, batch_size),
                     range(batch_size, 14000 + 1, batch_size))

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        start, end = training_samples[step%len(training_samples)]
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_x = np.array([s.reshape(128, -1)[:, 5:205] for s in samples[start: end]])# removing first 5 samples=~250ms
        batch_x = np.transpose(batch_x, [0, 2, 1])
        batch_y = classes[start:end]
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        if step % display_step == 0:
            # Calculate batch accuracy
            _, acc, loss = sess.run([optimizer, accuracy, cost_softmax],
                                    feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    n_length = 200
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_data = samples[14000:15000]
    test_data = np.array([np.resize(t, (n_input, n_length)).transpose() for t in test_data])
    test_label = classes[14000:15000]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label, keep_prob: 1.0}))
