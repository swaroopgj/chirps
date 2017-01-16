import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import gaussian_filter

batch_size = 128
test_size = 1000
beta = 0.0001

def prep_data(test_size=1000):
    #dataset1 = np.genfromtxt('fmel2828.txt', dtype=float, unpack=True)
    dataset1 = np.genfromtxt('featuresSG5656Great06000.txt', dtype=float, unpack=True)
    labelset1 = np.genfromtxt('labels0.txt', dtype=float, unpack=True)

    #dataset2 = np.genfromtxt('fmel28281.txt', dtype=float, unpack=True)
    dataset2 = np.genfromtxt('featuresSG5656Great8000.txt', dtype=float, unpack=True)
    labelset2 = np.genfromtxt('labels1.txt', dtype=float, unpack=True)

    data1 = dataset1.transpose()
    data2 = dataset2.transpose()
    print data1.shape, data2.shape

    #labs1 = np.zeros((6700, 2))
    labs1 = np.zeros((6000, 2))
    labs2 = np.zeros((8000, 2))

    for i in xrange(len(labs1)):

        if labelset1[i] == 0:
            labs1[i, 0] = 1
            labs1[i, 1] = 0

        else:
            labs1[i, 0] = 0
            labs1[i, 1] = 1

    for i in xrange(8000):

        if labelset2[i] == 0:
            labs2[i, 0] = 1
            labs2[i, 1] = 0
        else:
            labs2[i, 0] = 0
            labs2[i, 1] = 1

    data = np.concatenate((data1, data2), axis=0)
    print data.shape

    labs = np.concatenate((labs1, labs2), axis=0)
    # print labs
    ord = np.random.permutation(len(labs))
    # print ord
    data = data[ord]
    labs = labs[ord]
    datatr = data[0:13700]

    labstr = labs[0:13700]
    datate = data[13700:14700]
    labste = labs[13700:14700]
    '''
    dataaug=data[0:3000]
    labsaug=labs[0:3000]
    dataaug=gaussian_filter(dataaug, sigma=7)
    dataaug=np.add(dataaug,0.1*np.random.rand(3000,784))
    datatr=np.concatenate((dataaug, datatr))
    labstr=np.concatenate((labsaug,labstr))
    print datatr.shape,labstr.shape'''
    return datatr, labstr, datate, labste


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Conv Nets
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_maxpool(x, w_conv, b_conv):
    return max_pool2x2(tf.nn.relu(conv2d(x, w_conv) + b_conv))


def model(x, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2,
          keep_prob_conv, keep_prob_fc):
    # Conv-Pool layers
    h_conv1 = tf.nn.dropout(conv_maxpool(x, w_conv1, b_conv1), keep_prob_conv)
    h_conv2 = tf.nn.dropout(conv_maxpool(h_conv1, w_conv2, b_conv2), keep_prob_conv)
    h_conv3 = tf.nn.dropout(conv_maxpool(h_conv2, w_conv3, b_conv3), keep_prob_conv)
    # FC layers
    print(h_conv3.get_shape())
    h_conv3_flat = tf.reshape(h_conv3, [-1, w_fc1.get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_fc)
    # output
    return tf.matmul(h_fc1_drop, w_fc2) + b_fc2

'''
def model_old(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 28, 28, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,  # l2a shape=(?, 14, 14, 64)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx
'''

trX, trY, teX, teY = prep_data(test_size=1000)
#trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
#teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
trX = trX.reshape(-1, 56, 56, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 56, 56, 1)  # 28x28x1 input img

#X = tf.placeholder("float", [None, 28, 28, 1])
X = tf.placeholder("float", [None, 56, 56, 1])
Y = tf.placeholder("float", [None, 2])

w_conv1 = weight_variable(shape=[3, 3, 1, 32])
b_conv1 = bias_variable([32])
w_conv2 = weight_variable(shape=[3, 3, 32, 64])
b_conv2 = bias_variable([64])
w_conv3 = weight_variable(shape=[3, 3, 64, 128])
b_conv3 = bias_variable([128])
w_conv3 = weight_variable(shape=[3, 3, 64, 128])
b_conv3 = bias_variable([128])
#w_fc1 = weight_variable(shape=[128*4*4, 625])
w_fc1 = weight_variable(shape=[128*7*7, 625])
b_fc1 = bias_variable([625])
w_fc2 = weight_variable(shape=[625, 2])
b_fc2 = bias_variable([2])

'''
w = init_weights([3, 3, 1, 32])  # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])  # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625])  # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 2])  # FC 625 inputs, 10 outputs (labels)
'''

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
#py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
py_x = model(X, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2,
             p_keep_conv, p_keep_hidden)

cost_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
cost_l2 = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(w_conv3) + \
          tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(w_fc2)

cost = cost_softmax + beta*cost_l2

train_op = tf.train.RMSPropOptimizer(0.0005, 0.9).minimize(cost)  # init_lr=0.0001
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
cv_accs = []
loss_func = []
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(50):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))
        loss_epoch = []
        for start, end in training_batch:
            _, loss_iter = sess.run([train_op, cost_softmax],
                                    feed_dict={X: trX[start:end], Y: trY[start:end],
                                               p_keep_conv: 0.7, p_keep_hidden: 0.5})
            loss_epoch.append(loss_iter)
        loss_func.append(np.mean(loss_epoch))
        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        test_accuracy = np.mean(np.argmax(teY, axis=1) ==
                                sess.run(predict_op, feed_dict={X: teX,
                                                                p_keep_conv: 1.0,
                                                                p_keep_hidden: 1.0}))
        print(i, loss_func[-1], test_accuracy)
        cv_accs.append(test_accuracy)

#np.save('birds_test_run_accs.npy', cv_accs)
#np.save('birds_test_run_loss.npy', loss_func)
