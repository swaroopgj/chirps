
import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import gaussian_filter




dataset1=np.genfromtxt('fmel2828.txt',dtype=float, unpack=True)  
labelset1=np.genfromtxt('labels0.txt', dtype=float,unpack=True) 


dataset2=np.genfromtxt('fmel28281.txt',dtype=float, unpack=True)  
labelset2=np.genfromtxt('labels1.txt', dtype=float,unpack=True) 




data1= dataset1.transpose()
data2= dataset2.transpose()
print data1.shape, data2.shape



labs1=np.zeros((6700,2))
labs2=np.zeros((8000,2))

for i in xrange(6700):
       
        if labelset1[i]==0:
            labs1[i,0]=1
            labs1[i,1]=0
    
        else:
            labs1[i,0]=0
            labs1[i,1]=1
            
for i in xrange(8000):
   

    if labelset2[i]==0:
        labs2[i,0]=1
        labs2[i,1]=0
    else:
        labs2[i,0]=0
        labs2[i,1]=1



data=np.concatenate((data1, data2), axis=0)
print data.shape


labs=np.concatenate((labs1, labs2), axis=0)
#print labs



ord=np.random.permutation(14700)
#print ord
data=data[ord]
labs=labs[ord]
datatr=data[0:13700]

labstr=labs[0:13700]
datate=data[13700:14700]
labste=labs[13700:14700]
'''
dataaug=data[0:3000]
labsaug=labs[0:3000]
dataaug=gaussian_filter(dataaug, sigma=7)
dataaug=np.add(dataaug,0.1*np.random.rand(3000,784))


datatr=np.concatenate((dataaug, datatr))
labstr=np.concatenate((labsaug,labstr))

print datatr.shape,labstr.shape'''

batch_size = 128
test_size=1000

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)


    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


trX, trY, teX, teY = datatr, labstr, datate, labste
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 2])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])    # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 2])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
cost_l2 = tf.nn.l2_loss()
train_op = tf.train.RMSPropOptimizer(0.0005, 0.9).minimize(cost) #init_lr=0.0001
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
cv_accs = []
loss_func = []
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(200):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            _, loss_iter = sess.run([train_op, cost], feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.7, p_keep_hidden: 0.5})
            loss_func.append(loss_iter)
        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        test_accuracy = np.mean(np.argmax(teY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: teX,
                                                             p_keep_conv: 1.0, p_keep_hidden: 1.0}))
        print(i, test_accuracy)
        cv_accs.append(test_accuracy)

np.save('birds_test_run_accs.npy', cv_accs)
np.save('birds_test_run_loss.npy', loss_func)
