# I implemented my ConvNets going through the Tensorflow Deep MNIST tutorial
# Apparently, I ended up getting my accuracy around 78%. I think my accuracy could have been higher
# if I tried more diverse data sets such as cropped or distorted images rather than just random sorting the dataset.

# My ConvNets architecture

# CONV_1(CONV2D, RELU) -> POOL1 -> CONV_2(CONV2D, RELU) -> POOL2 -> CONV_3(CONV2D, RELU) ->
# CONV_4(CONV2D, RELU) -> CONV5(CONV2D, RELU) -> FC1(DENSE) -> DROPOUT -> FC2(READOUT) -> SOFTMAX
# Conv2D                 : strides of 1 and are zero padded
# RELU                   : activation function, max(0, x), thresholding at 0
# pooling                : max pooling to downsample
# FC1(dense)             : fully connected layer of 384 neurons to allow process the image
# dropout                : layer to prevent overfitting
# FC2(readout)           : fully connected layer that maps 384 features into the 10 cifar10 labels
# Softmax                : classifier to determine the label
import os
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

def cifar10_cnn(x):
    # conv1: 64 features, 5 x 5 patch
    with tf.name_scope('conv1'):
        w_conv1 = weight([5, 5, 3, 64])
        b_conv1 = bias([64])
        h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)

    # pool1: 2 x 2 blocks
    with tf.name_scope('pool1'):
        h_pool1 = max_pool(h_conv1)

    # conv2: 64 features, 5 x 5 patch
    with tf.name_scope('conv2'):
        w_conv2 = weight([5, 5, 64, 64])
        b_conv2 = bias([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    # pool2: 2 x 2 blocks
    with tf.name_scope('pool2'):
        h_pool2 = max_pool(h_conv2)

    # conv3: 128 features, 3 x 3 patch
    with tf.name_scope('conv3'):
        w_conv3 = weight([3, 3, 64, 128])
        b_conv3 = bias([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)

    # conv4: 128 features, 3 x 3 patch
    with tf.name_scope('conv4'):
        w_conv4 = weight([3, 3, 128, 128])
        b_conv4 = bias([128])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, w_conv4) + b_conv4)

    # conv5: 128 features, 3 x 3 patch
    with tf.name_scope('conv5'):
        w_conv5 = weight([3, 3, 128, 128])
        b_conv5 = bias([128])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, w_conv5) + b_conv5)

    # FC1(Dense): layer that maps to 384 features
    with tf.name_scope('fc1'):
        w_fc1 = weight([8 * 8 * 128, 384])
        b_fc1 = bias([384])
        h_flat = tf.reshape(h_conv5, [-1, 8 * 8 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, w_fc1) + b_fc1)

    # Dropout layer: layer to prevent the overfitting of the graph
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2(Readout) : layer that maps the features back to 10 cifar10 classes
    with tf.name_scope('fc2'):
        w_fc2 = weight([384, 10])
        b_fc2 = bias([10])
        out = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    return out, keep_prob

# conv layer
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# pooling layer
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');

# weight initialization
def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# bias initialization
def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main():
    # load data using Keras and forget about the data helper
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # print "x_train: ", x_train.shape  # x_train:  (50000, 32, 32, 3)
    # print "y_train: ", y_train.shape  # y_train:  (50000, 1)
    # print "x_test: ", x_test.shape    # x_test:  (10000, 32, 32, 3)
    # print "y_test: ", y_test.shape    # y_test:  (10000, 1)

    # keras dataset have y as a scalar value, so use one hot encoding
    y_train_ = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
    y_test_ = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

    # placeholders for training data and labels
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, 10])

    # create the cnn graph
    out, keep_prob = cifar10_cnn(x)

    # loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

    #calculate the accuracy of the graph
    pred = tf.argmax(out, 1)
    target = tf.argmax(y, 1)

    correct_pred = tf.equal(pred, target)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    batch_size = 150                        # whatever batch size I want
    epoch = 300                             # 1 epoch = 1 cycle of training data will run
    steps = int(len(x_train) / batch_size)  # number of steps to complete a single epoch
    saver = tf.train.Saver()
    if not os.path.exists("model/"):
        os.makedirs("model/")
    saver = tf.train.Saver()
    saver_path = "./model/cifar10.ckpt"
    with tf.Session() as sess:
        # initialize tensor variables
        init = tf.global_variables_initializer()
        sess.run(init)
        # training cycle
        for i in range(epoch):
            # train batches
            for j in range(steps):
                # create a batch from train data set using a random number generator
                random_int = np.random.randint(0, len(x_train), size=batch_size)
                y_batch = y_train_.eval()
                x_train_batch = np.asarray([x_train[k] for k in random_int])
                y_train_batch = np.asarray([y_batch[k] for k in random_int])
                sess.run(optimizer, feed_dict={x: x_train_batch, y: y_train_batch, keep_prob: 0.5})
            
            train_accuracy, train_loss = sess.run([accuracy, loss], feed_dict={x: x_train_batch, y: y_train_batch, keep_prob: 1.0}) 
            #print evaluation for each epoch
            print("Epoch %d, Acurracy: %.3f, Loss: %.3f" % (i, train_accuracy, train_loss))

            # save the graph for every 100 epoch
            if (i % 100 == 0) or (i == epoch - 1):
                 saver.save(sess, saver_path)

        # test the result with test data
        print "Test Accuracy: %.3f" % accuracy.eval(feed_dict={x: x_test, y: y_test_.eval(), keep_prob:1.0})
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, saver_path)
    
        print "Final Test accuracy: %.3f" % accuracy.eval(feed_dict={x: x_test, y:y_test_.eval(), keep_prob:1.0})

if __name__ == '__main__':
    main()
