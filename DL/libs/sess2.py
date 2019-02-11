import tensorflow as tf
import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np

n_observatons = 1000
xs = np.linspace(-3.0, 3.0, n_observatons)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observatons)
print(xs.shape, ys.shape)

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

def distance(d1, d2):
    return tf.abs(d1 - d2)

fig, ax = plt.subplots(1, 1)
ax.scatter(xs, ys, alpha=0.15, marker="+")

def d1_weight():
    W = tf.Variable(tf.random_normal([1], dtype=tf.float32, stddev=0.1), name="weight")
    B = tf.Variable(tf.constant([0], dtype=tf.float32), name="bias")

    Y_pred = X * W + B 
    train(X, Y, Y_pred)

def dn_weight():
    n_neurons = 100
    W = tf.Variable(tf.random_normal([1, n_neurons], stddev=0.1))
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_neurons]))
    h = tf.matmul(tf.expand_dims(X, 1), W) + b
    Y_pred = tf.reduce_sum(h, 1)
    train(X, Y, Y_pred)

def activation_tanh():
    n_neurons = 10
    W = tf.Variable(tf.random_normal([1, n_neurons]), name="W")
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_neurons]), name="b")
    h = tf.nn.tanh(tf.matmul(tf.expand_dims(X, 1), W) + b)
    Y_pred = tf.reduce_sum(h, 1)
    train(X, Y, Y_pred)

def activation_sigmoid():
    n_neurons = 10
    W = tf.Variable(tf.random_normal([1, n_neurons]), name="W")
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_neurons]), name="b")
    h = tf.nn.sigmoid(tf.matmul(tf.expand_dims(X, 1), W) + b)
    Y_pred = tf.reduce_sum(h, 1)
    train(X, Y, Y_pred)
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])

def activation_relu():
    n_neurons = 10
    W = tf.Variable(tf.random_normal([1, n_neurons]), name="W")
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_neurons]), name="b")
    h = tf.nn.relu(tf.matmul(tf.expand_dims(X, 1), W) + b)
    Y_pred = tf.reduce_sum(h, 1)
    train(X, Y, Y_pred)
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])

def linear(X, n_input, n_output, activation=None):
    W = tf.Variable(tf.random_normal([n_input, n_output]), name="W")
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_output]), name="b")
    h = tf.nn.tanh(tf.matmul(tf.expand_dims(X, 1), W) + b)
    return h

def linear_improve(X, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable(name="W", shape=[n_input, n_output], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(name="b", shape=[n_output], initializer=tf.constant_initializer())
        h = tf.matmul(X, W) + b
        if activation is not None:
            h = activation(h)
        return h

def build_layers(X):
    n_neurons = [2, 64, 64, 64, 64, 64, 64, 3]
    current_input = X
    for layer_i in range(1, len(n_neurons)):
        current_input = linear_improve(
            X=current_input,
            n_input=n_neurons[layer_i -1],
            n_output=n_neurons[layer_i],
            activation=tf.nn.relu if (layer_i+1) < len(n_neurons) else None,
            scope="layer_" + str(layer_i))

    return current_input

def train_painting(X, Y, Y_pred, img, n_iterations=500, batch_size=50, learning_rate=0.001)):
    cost = tf.reduct_mean(tf.reduce_sum(distance(Y_pred, Y), 1))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for it_i in range(n_iterations):
            idxs = np.random.permutation(len(xs))
            n_batches = len(idxs) // batch_size
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size : (batch_i + 1)* batch_size]
                sess.run(optimizer, feed_dict={X:xs[idxs_i], Y:ys[idxs_i]})

            training_cost = sess.run(cost, feed_dict={X:xs, Y:ys})

            if it_i % 20 == 0:
                ys_pred = Y_pred.eval(feed_dict={X:xs}, session=sess)
                fig, ax = plt.subplots(1, 1)
                img_out = np.clip(ys_pred.reshape(img.shape), 0, 255).astype(np.uint8)
                plt.imshow(img_out)
                plt.show()


def graph_flow():
    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    g = tf.get_default_graph()
    opes = [op.name for op in tf.get_default_graph().get_operations()]
    print(opes)

    X = tf.placeholder(tf.float32, name="X")
    h = linear_improve(X, 2, 10)
    h2 = linear_improve(X, 3, 12, scope='linear2')
    h3 = linear_improve(X, 3, 12, scope='linear3')

    # h = linear(X, 2, 10)
    # h2 = linear(X, 3, 12)
    # h3 = linear(X, 3, 12)

    opes = [op.name for op in tf.get_default_graph().get_operations()]
    print(opes)


def train(X, Y, Y_pred, n_iterations=1000, batch_size=200, learning_rate=0.02):
    cost = tf.reduce_mean(distance(Y_pred, Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for it_i in range(n_iterations):
            idxs = np.random.permutation(len(xs))
            n_batches = len(idxs) // batch_size
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size : (batch_i + 1)* batch_size]
                sess.run(optimizer, feed_dict={X:xs[idxs_i], Y:ys[idxs_i]})

            training_cost = sess.run(cost, feed_dict={X:xs, Y:ys})

            if it_i % 10 == 0:
                ys_pred = Y_pred.eval(feed_dict={X:xs}, session=sess)
                ax.plot(xs, ys_pred, 'k', alpha=it_i / n_iterations)
                print(training_cost)
    # fig.show()
    plt.show()

# example code from homework

def split_image(img):
    # We'll first collect all the positions in the image in our list, xs
    xs = []

    # And the corresponding colors for each of these positions
    ys = []

    # Now loop over the image
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            # And store the inputs
            xs.append([row_i, col_i])
            # And outputs that the network needs to learn to predict
            ys.append(img[row_i, col_i])

    # we'll convert our lists to arrays
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def build_model(xs, ys, n_neurons, n_layers, activation_fn,
                final_activation_fn, cost_type):
    
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    
    if xs.ndim != 2:
        raise ValueError(
            'xs should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')
    if ys.ndim != 2:
        raise ValueError(
            'ys should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')
        
    n_xs = xs.shape[1]
    n_ys = ys.shape[1]
    
    X = tf.placeholder(name='X', shape=[None, n_xs],
                       dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=[None, n_ys],
                       dtype=tf.float32)

    current_input = X
    for layer_i in range(n_layers):
        current_input = utils.linear(
            current_input, n_neurons,
            activation=activation_fn,
            name='layer{}'.format(layer_i))[0]

    Y_pred = utils.linear(
        current_input, n_ys,
        activation=final_activation_fn,
        name='pred')[0]
    
    if cost_type == 'l1_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.abs(Y - Y_pred), 1))
    elif cost_type == 'l2_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.squared_difference(Y, Y_pred), 1))
    else:
        raise ValueError(
            'Unknown cost_type: {}.  '.format(
            cost_type) + 'Use only "l1_norm" or "l2_norm"')
    
    return {'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost}

def train(imgs,
          learning_rate=0.0001,
          batch_size=200,
          n_iterations=100,
          gif_step=2,
          n_neurons=30,
          n_layers=10,
          activation_fn=tf.nn.relu,
          final_activation_fn=tf.nn.tanh,
          cost_type='l2_norm'):

    N, H, W, C = imgs.shape
    all_xs, all_ys = [], []
    for img_i, img in enumerate(imgs):
        xs, ys = split_image(img)
        all_xs.append(np.c_[xs, np.repeat(img_i, [xs.shape[0]])])
        all_ys.append(ys)
    xs = np.array(all_xs).reshape(-1, 3)
    xs = (xs - np.mean(xs, 0)) / np.std(xs, 0)
    ys = np.array(all_ys).reshape(-1, 3)
    ys = ys / 127.5 - 1

    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model = build_model(xs, ys, n_neurons, n_layers,
                            activation_fn, final_activation_fn,
                            cost_type)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(model['cost'])
        sess.run(tf.global_variables_initializer())
        gifs = []
        costs = []
        step_i = 0
        for it_i in range(n_iterations):
            # Get a random sampling of the dataset
            idxs = np.random.permutation(range(len(xs)))

            # The number of batches we have to iterate over
            n_batches = len(idxs) // batch_size
            training_cost = 0

            # Now iterate over our stochastic minibatches:
            for batch_i in range(n_batches):

                # Get just minibatch amount of data
                idxs_i = idxs[batch_i * batch_size:
                              (batch_i + 1) * batch_size]

                # And optimize, also returning the cost so we can monitor
                # how our optimization is doing.
                cost = sess.run(
                    [model['cost'], optimizer],
                    feed_dict={model['X']: xs[idxs_i],
                               model['Y']: ys[idxs_i]})[0]
                training_cost += cost

            print('iteration {}/{}: cost {}'.format(
                    it_i + 1, n_iterations, training_cost / n_batches))

            # Also, every 20 iterations, we'll draw the prediction of our
            # input xs, which should try to recreate our image!
            if (it_i + 1) % gif_step == 0:
                costs.append(training_cost / n_batches)
                ys_pred = model['Y_pred'].eval(
                    feed_dict={model['X']: xs}, session=sess)
                img = ys_pred.reshape(imgs.shape)
                gifs.append(img)
        return gifs

if __name__ == "__main__":
    # d1_weight()
    # dn_weight()
    # activation_relu()
    # activation_tanh()
    graph_flow()
