import tensorflow as tf
import numpy as np

batch_size = 128
test_size = 1000
beta = 0.0001
tf.reset_default_graph()


def prep_data(mfcc=False, logamp=True, seed=1024):
    import cPickle as pickle
    data_fn = 'song_data_mfcc.pkl' if mfcc else 'song_data_melspec.pkl'
    print "Using %s" % data_fn
    with open(data_fn, 'r') as f:
        data = pickle.load(f)
    samples = data['data']
    classes = [data['labels'][k] for k in data['names']]
    classes = np.asarray([[1, 0] if l == 0 else [0, 1] for l in classes])
    # remove samples <205 and >250
    shapes = np.array([s.shape[1] / 128 for s in samples])
    filter = (shapes < 250) & (shapes > 205)
    samples = [samples[i] for i in range(len(samples)) if filter[i]]
    classes = classes[filter]
    # clean data by replacing zeros with min() in each sample
    if not mfcc and logamp:
        for isamp in range(len(samples)):
            samples[isamp][samples[isamp] == 0] = samples[isamp][samples[isamp] > 0].min()
            samples[isamp] = np.log10(samples[isamp])
            samples[isamp] -= np.median(samples[isamp])
    # permutate
    np.random.seed(seed)
    ord = np.random.permutation(len(samples))
    samples = [samples[i] for i in ord]
    classes = classes[ord]
    # using only high frequencies
    if mfcc:
        samples = np.array([s.reshape(128, -1)[:20, 5:205].reshape(-1) for s in samples])
    else:
        samples = np.array([s.reshape(128, -1)[64:, 5:205].reshape(-1) for s in samples])
    del data
    datatr = samples[0:14600]
    labstr = classes[0:14600]
    datate = samples[14600:]
    labste = classes[14600:]
    return datatr, labstr, datate, labste


def prep_test_data(mfcc=False, logamp=True):
    import cPickle as pickle
    data_fn = 'test_data_mfcc.pkl' if mfcc else 'test_data_melspec.pkl'
    nfeatures = 20 if mfcc else 128
    print "Using %s" % data_fn
    with open(data_fn, 'r') as f:
        data = pickle.load(f)
    samples = data['data']
    fnames = data['names']
    # remove samples <205 and >250
    shapes = np.array([s.shape[1] / nfeatures for s in samples])
    #filter = (shapes < 250) & (shapes > 205)
    #samples = [samples[i] for i in range(len(samples)) if filter[i]]
    #fnames = [fnames[i] for i in range(len(samples)) if filter[i]]
    # clean data by replacing zeros with min() in each sample
    if not mfcc and logamp:
        for isamp in range(len(samples)):
            samples[isamp][samples[isamp] == 0] = samples[isamp][samples[isamp] > 0].min()
            samples[isamp] = np.log10(samples[isamp])
            samples[isamp] -= np.median(samples[isamp])
    #samples = np.array([s.reshape(20, -1)[:, :200].reshape(-1) for s in samples])
    extended_samples = []
    extended_fnames = []
    for i in range(len(samples)):
        if shapes[i] > 200:
            samples[i] = samples[i].reshape(nfeatures, -1)[:, 5:205].reshape(-1)
            if shapes[i] > 217:
                extended_samples.append(samples[i].reshape(nfeatures, -1)[:, 205:].reshape(-1))
                extended_fnames.append(fnames[i])
    samples.extend(extended_samples)
    fnames.extend(extended_fnames)
    if mfcc:
        samples = np.array([np.resize(s.reshape(20, -1).T, (200, 20)).T.reshape(-1) for s in samples])
    else:
        samples = np.array(
            [np.resize(s.reshape(128, -1).T, (200, 128)).T[64:, :].reshape(-1) for s in samples])
    del data
    return samples, fnames

# Final submission
# combine_subs(['sub1_melspecmodel1-33.csv','sub1_melspecmodel0-30.csv','sub1_mfccmodel2-40.csv'])
def combine_subs(flist):
    fnames = []
    vals = []
    for f in flist:
        with open(f, 'r') as file:
            text = file.readlines()
        fnames.append([t.split(',')[0] for t in text])
        vals.append([float(t.split(',')[1].strip()) for t in text])
    vals = np.asarray(vals)
    vals = np.max(vals, axis=0)
    vals[vals<0.51] = 0.0
    with open('sub1.csv', 'r') as file:
        text = file.readlines()
    final_fnames = [t.split(',')[0] for t in text]
    final_probs = np.array([float(t.split(',')[1].strip()) for t in text])
    final_probs = np.vstack([final_probs, vals])
    final_probs = final_probs.max(axis=0)
    with open('sub1_combined_new.csv', 'w') as subfile:
        for i in range(len(final_fnames)):
            subfile.write("%s,%f\n" % (final_fnames[i], final_probs[i]))


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
    print(h_conv3.get_shape())
    h_conv3_flat = tf.reshape(h_conv3, [-1, w_fc1.get_shape().as_list()[0]])
    print(h_conv3_flat.get_shape())
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)
    print(h_fc1.get_shape())
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_fc)
    # output
    return tf.matmul(h_fc1_drop, w_fc2) + b_fc2


if __name__ == "__main__":
    X = tf.placeholder("float", [None, 64, 200, 1])
    Y = tf.placeholder("float", [None, 2])

    w_conv1 = weight_variable(shape=[3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    w_conv2 = weight_variable(shape=[3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    w_conv3 = weight_variable(shape=[3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    w_conv3 = weight_variable(shape=[3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    w_fc1 = weight_variable(shape=[8*8*128, 512])
    b_fc1 = bias_variable([512])
    w_fc2 = weight_variable(shape=[512, 2])
    b_fc2 = bias_variable([2])

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
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
        # you need to initialize all variables
        saver = tf.train.Saver(max_to_keep=20)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        # training data
        trX, trY, teX, teY = prep_data()
        trX = trX.reshape(-1, 64, 200, 1)
        teX = teX.reshape(-1, 64, 200, 1)
        saved_model_counter = 0
        for i in range(50):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX) + 1, batch_size))
            loss_epoch = []
            for start, end in training_batch:
                _, loss_iter = sess.run([train_op, cost_softmax],
                                        feed_dict={X: trX[start:end], Y: trY[start:end],
                                                   p_keep_conv: 0.5, p_keep_hidden: 0.5})
                loss_epoch.append(loss_iter)
            loss_func.append(np.mean(loss_epoch))
            #test_indices = np.arange(len(teX))  # Get A Test Batch
            #np.random.shuffle(test_indices)
            #test_indices = test_indices[0:test_size]

            test_accuracy = np.mean(np.argmax(teY, axis=1) ==
                                    sess.run(predict_op, feed_dict={X: teX,
                                                                    p_keep_conv: 1.0,
                                                                    p_keep_hidden: 1.0}))
            print(i, loss_func[-1], test_accuracy)
            if test_accuracy > 0.855:
                saver.save(sess, './melspecmodels/melspec_model1', global_step=saved_model_counter)
                print("saved model %d: %f" % (saved_model_counter, test_accuracy))
                saved_model_counter += 1

            cv_accs.append(test_accuracy)

        #saver.save(sess, 'cnn_melspec_model0')
    # Restore sess
    testing = True
    model_fname = 'melspecmodels/saved/melspec_model1-33'
    if testing:
        sess = tf.Session()
        # sess.run(tf.initialize_all_variables())
        # new_saver = tf.train.import_meta_graph('mfccmodels/cnn_melspec_model0.meta')
        new_saver = tf.train.Saver()
        new_saver.restore(sess, model_fname)
        testX, fnames = prep_test_data(mfcc=False)
        testX = testX.reshape(-1, 64, 200, 1)
        logits = np.asarray([sess.run(py_x,
                                      feed_dict={X: testX[i, ][None, ], p_keep_conv: 1.0,
                                                 p_keep_hidden: 1.0})
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
        with open('sub1_melspecmodel1-33.csv','w') as subfile:
            for i in range(len(final_fnames)):
                subfile.write("%s,%f\n"%(final_fnames[i].split('.wav')[0], final_probs[i]))
        '''

    # Restore sess
    '''
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    new_saver = tf.train.import_meta_graph('melspecmodels/cnn_melspec_model0.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./melspecmodels/'))

    teX, fnames = prep_test_data()
    teX = teX.reshape(-1, 64, 200, 1)
    logits = np.asarray([sess.run(py_x, feed_dict={X: teX[i,][None,], p_keep_conv: 1.0, p_keep_hidden: 1.0})
                for i in range(len(teX))]).squeeze()
    probs = tf.nn.softmax(logits)
    test_probs = sess.run(probs)
    with open('sub1.csv','w') as subfile:
        for i in range(len(fnames)):
            subfile.write("%s,%f\n"%(fnames[i].split('.wav')[0], test_probs[i,1]))

    '''
    #np.save('birds_test_run_accs.npy', cv_accs)
    #np.save('birds_test_run_loss.npy', loss_func)
