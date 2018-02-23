import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os


mnist = input_data.read_data_sets("./mnist", one_hot=True) # import data
x_train = mnist.train.images  # image data for training
y_train = mnist.train.labels  # image data labels for training correspondingly
x_test = mnist.test.images  # image data for testing
y_test = mnist.test.labels  # image data labels for testing correspondingly

print "x_train: ", x_train.shape  # shape of training data
print "y_train: ", y_train.shape  # shape of testing data labels
print "x_test: ", x_test.shape  # shape of testing data
print "y_test: ", y_test.shape  # shape of testing data labels


def plot_mnist(data, classes):
    for i in range(10):
        idxs = (classes == i)  # match images to the class i
        images = data[idxs][0:10]  # get 10 images for class i

        for j in range(5):
            plt.subplot(5, 10, i + j * 10 + 1)
            plt.imshow(images[j].reshape(28, 28), cmap='gray')  # change the shape back to 28*28
            if j == 0:  # print a title only once for each class
                plt.title(i)
            plt.axis('off')
    plt.show()


classes = np.argmax(y_train, 1)
print classes
plot_mnist(x_train, classes)  # plot examples


def fully_connected(x, dim_in, dim_out, name):
    with tf.variable_scope(name) as scope:
        # create variables
        w = tf.get_variable('w', shape=[dim_in, dim_out],
                            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        b = tf.get_variable('b', shape=[dim_out])

        # create operations
        out = tf.matmul(x, w) + b

        return out

# Create model
# 256 features(whatever number of features i wish...)
def neural_network(x, dim_in=784, dim_h=256, dim_out=10):

    h1 = fully_connected(x, dim_in, dim_h, name='h1')  # 1st hidden layer with ReLU
    h1 = tf.nn.relu(h1)  # output from the 1st hidden layer

    h2 = fully_connected(h1, dim_h, dim_h, name='h2')  # 2nd hidden layer with ReLU
    h2 = tf.nn.relu(h2)  # output from the 2nd hidden layer

    out = fully_connected(h2, dim_h, dim_out, name='out')  # classification labels

    return out

x = tf.placeholder(tf.float32, [None, 784])  # create placeholders for the training data
y = tf.placeholder(tf.float32, [None, 10])  # create placeholders for the training data labels

# Construct model with default value
out = neural_network(x)

# loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# Test model
pred = tf.argmax(out, 1)
target = tf.argmax(y, 1)

correct_pred = tf.equal(pred, target)
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

batch_size = 100  # define the number of data in training for updates
save_every = 1  # save the model for each iteration of training

if not os.path.exists('model/'):
    os.makedirs('model/')

# launch the graph
with tf.Session() as sess:
    # initialize tensor variables
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=20)
    # training cycle
    for epoch in range(10000):
        avg_loss = 0.0
        n_updates_per_epoch = int(mnist.train.num_examples / batch_size)
        # loop over all batches
        for i in range(n_updates_per_epoch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            # run optimization op (backprop) and loss op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch})
            # compute average loss
            avg_loss += c / n_updates_per_epoch
        print "Epoch %d, Loss: %.3f" % (epoch + 1, avg_loss)
        if epoch % save_every == 0:
            saver.save(sess, save_path='model/fc', global_step=epoch + 1)

    print "\nTest accuracy:", sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels})

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'model/fc-1')

    print "\nTest accuracy:", sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels})
