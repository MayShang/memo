import tensorflow as tf
import numpy as np

#create or prepare training data
trainX = np.linspace(-1, 1, 100)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33

# x, y placeholder to train
X = tf.placeholder(dtype=tf.float32, name="X")
Y = tf.placeholder(dtype=tf.float32, name="Y")

# modeling: how to calc Y_pred
w = tf.Variable(0.0, name="w")
y_model = tf.multiply(X, w)

# cost and training
cost = (tf.pow(y_model - Y, 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cost_t = 0
    for i in range(1000):
        for (x, y) in zip(trainX, trainY):
            _, c = sess.run([train_op, cost], feed_dict={X:x, Y:y})
            cost_t += c
    print("cost is {:.9f}".format(cost_t/1000))
    print(sess.run(w))
