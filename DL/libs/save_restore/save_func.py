from __future__ import print_function

import numpy as np
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "./save_func/model.ckpt"

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

with tf.name_scope('weights_variables'):
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="h1"),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='out')
    }

with tf.name_scope('biases_variables'):
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
        'out': tf.Variable(tf.random_normal([n_classes]), name='out')
    }

# Construct model
with tf.name_scope('OP_pred'):
    pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
with tf.name_scope('Op_cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

with tf.name_scope('Op_optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
with tf.name_scope('Op_init'):
    init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

def save_model():
    # Running first session
    print("Starting 1st session...")
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training cycle
        for epoch in range(3):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("First Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        # Save model weights to disk
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

def restore_model():
    # Running a new session
    print("Starting 2nd session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        # print("Model restored from file: %s" % save_path)
        print("Model restored from file")

        # Resume training
        for epoch in range(7):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Second Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval(
            {x: mnist.test.images, y: mnist.test.labels}))

def restore_model_x():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./save_func/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./save_func/'))
    g = tf.get_default_graph()
    names = [op.name for op in g.get_operations()]
    print(names[:70])

if __name__ == '__main__':
    # save_model()
    restore_model()
    # restore_model_x()
