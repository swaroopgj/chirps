import tensorflow as tf
import numpy as np
from cnn_melspec import prep_data, prep_test_data
batch_size = 128
test_size = 1000
beta = 0.0001
tf.reset_default_graph()
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
    return tf.nn.max_pool(x, ksize=[1, 2, 4, 1], strides=[1, 2, 3, 1], padding='SAME')


def conv_maxpool(x, w_conv, b_conv):
    return max_pool2x2(tf.nn.relu(conv2d(x, w_conv) + b_conv))


def model(x, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2,
          keep_prob_conv, keep_prob_fc):
    # Conv-Pool layers
    h_conv1 = tf.nn.dropout(conv_maxpool(x, w_conv1, b_conv1), keep_prob_conv)
    h_conv2 = tf.nn.dropout(conv_maxpool(h_conv1, w_conv2, b_conv2), keep_prob_conv)
    h_conv3 = tf.nn.dropout(conv_maxpool(h_conv2, w_conv3, b_conv3), keep_prob_conv)
    # FC layers
    #print(h_conv3.get_shape())
    h_conv3_flat = tf.reshape(h_conv3, [-1, w_fc1.get_shape().as_list()[0]])
    #print(h_conv3_flat.get_shape())
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)
    #print(h_fc1.get_shape())
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_fc)
    # output
    return tf.matmul(h_fc1_drop, w_fc2) + b_fc2


X = tf.placeholder("float", [None, 20, 200, 1])
Y = tf.placeholder("float", [None, 2])

w_conv1 = weight_variable(shape=[3, 3, 1, 32])
b_conv1 = bias_variable([32])
w_conv2 = weight_variable(shape=[3, 3, 32, 64])
b_conv2 = bias_variable([64])
w_conv3 = weight_variable(shape=[3, 3, 64, 128])
b_conv3 = bias_variable([128])
w_conv3 = weight_variable(shape=[3, 3, 64, 128])
b_conv3 = bias_variable([128])
w_fc1 = weight_variable(shape=[3*8*128, 512])
b_fc1 = bias_variable([512])
w_fc2 = weight_variable(shape=[512, 2])
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
#print(py_x)
cost_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
cost_l2 = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(w_conv3) + \
          tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(w_fc2)

cost = cost_softmax + beta*cost_l2

train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)  # init_lr=0.0001
predict_op = tf.argmax(py_x, 1)

training = False
if training:
    # Launch the graph in a session
    cv_accs = []
    loss_func = []
    trX, trY, teX, teY = prep_data(mfcc=True, seed=1729)
    trX = trX.reshape(-1, 20, 200, 1)
    teX = teX.reshape(-1, 20, 200, 1)

    #with tf.Session() as sess:
    # you need to initialize all variables
    saver = tf.train.Saver(max_to_keep=20)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    saved_model_counter = 0
    for i in range(100, 150):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))
        loss_epoch = []
        for start, end in training_batch:
            _, loss_iter = sess.run([train_op, cost_softmax],
                                    feed_dict={X: trX[start:end], Y: trY[start:end],
                                               p_keep_conv: 0.5, p_keep_hidden: 0.5})
            loss_epoch.append(loss_iter)
        loss_func.append(np.mean(loss_epoch))
        test_accuracy = np.mean(np.argmax(teY, axis=1) ==
                                sess.run(predict_op, feed_dict={X: teX,
                                                                p_keep_conv: 1.0,
                                                                p_keep_hidden: 1.0}))
        print(i, loss_func[-1], test_accuracy)
        if test_accuracy > 0.865:
            saver.save(sess, './mfccmodels/mfcc_model1', global_step=saved_model_counter)
            print("saved model %d: %f" % (saved_model_counter, test_accuracy))
            saved_model_counter += 1
        cv_accs.append(test_accuracy)

#saver.save(sess, 'cnn_mfcc_model0')
# Restore sess
testing = True
#model_fname = 'mfccmodels/firstsub/mfcc_model-1000'
model_fname = 'mfccmodels/model2/mfcc_model1-40'
if testing:
    sess = tf.Session()
    #sess.run(tf.initialize_all_variables())
    #new_saver = tf.train.import_meta_graph('mfccmodels/cnn_melspec_model0.meta')
    new_saver = tf.train.Saver()
    new_saver.restore(sess, model_fname)
    testX, fnames = prep_test_data(mfcc=True)
    testX = testX.reshape(-1, 20, 200, 1)
    logits = np.asarray([sess.run(py_x, feed_dict={X: testX[i,][None,], p_keep_conv: 1.0, p_keep_hidden: 1.0})
                for i in range(len(testX))]).squeeze()
    probs = tf.nn.softmax(logits)
    test_probs = sess.run(probs)
    test_probs = test_probs[:, 1]
    # final probs
    final_probs = []
    final_fnames = []
    for i, f in enumerate(fnames):
        if f not in final_fnames:
            final_fnames.append(f)
            final_probs.append(test_probs[i])
        else:
            final_probs[final_fnames.index(f)] = max(final_probs[final_fnames.index(f)],
                                                     test_probs[i])
    final_probs = np.array(final_probs)
    '''
    with open('sub1_mfccmodel2-40.csv','w') as subfile:
        for i in range(len(final_fnames)):
            subfile.write("%s,%f\n"%(final_fnames[i].split('.wav')[0], final_probs[i]))
    '''

'''
sess = tf.Session()
new_saver = tf.train.import_meta_graph('my-model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.trainable_variables()
'''
#np.save('birds_test_run_accs.npy', cv_accs)
#np.save('birds_test_run_loss.npy', loss_func)
