import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from libs import gif, nb_utils
from libs import inception
from libs import vgg16
import os

def load_inception_model():
    net = inception.get_inception_model()

def load_vgg16_model():
    net = vgg16.get_vgg_model()
    g = tf.Graph()
    with tf.Session(graph=g) as sess, g.device('/cpu:0'):
        tf.import_graph_def(net['graph_def'], name='vgg')
        names = [op.name for op in g.get_operations()]
    return net, names, g

def load_file(fname):
    return plt.imread(fname)

def load_img_from_skimg():
    from skimage.data import astronaut
    img = astronaut()
    # img = np.array(img)
    return img

def eval_tensor(g, net, tensor, placeholder, img, dropout_list):
    with tf.Session(graph=g) as sess, g.device('/cpu:0'):
        res = tensor.eval(feed_dict={
            placeholder: img,
            dropout_list[0][0]+':0' : [[1.0] * dropout_list[1][0]],
            dropout_list[0][1] + ':0' : [[1.0] * dropout_list[1][1]]})[0]
        reses = [(res[idx], net['labels'][idx]) for idx in res.argsort()[-5:][::-1]]
        return reses

def get_tensor_eval(g, layer_name, placeholder, placeholder_content, dropout_list):
    with tf.Session(graph=g) as sess, g.device('/cpu:0'):
        tensor_val = g.get_tensor_by_name(layer_name).eval(
            session=sess,
            feed_dict={
                placeholder : placeholder_content,
                dropout_list[0][0]+':0' : [[1.0] * dropout_list[1][0]],
                dropout_list[0][1] + ':0' : [[1.0] * dropout_list[1][1]]
            })
        return tensor_val

def get_tensor_evals(g, layer_names, placeholder, placeholder_content, dropout_list):
    """
    for the input, we can apply speicific layer to it and get to know input every
    pixel value at this layer.
    """

    tensor_vals = []
    with tf.Session(graph=g) as sess, g.device('/cpu:0'):
        for layer_name_i in layer_names:
            tensor_val = g.get_tensor_by_name(layer_name_i).eval(
                session=sess,
                feed_dict={
                    placeholder : placeholder_content,
                    dropout_list[0][0]+':0' : [[1.0] * dropout_list[1][0]],
                    dropout_list[0][1] + ':0' : [[1.0] * dropout_list[1][1]]
                })
            tensor_vals.append(tensor_val)
    return tensor_vals

def get_content_loss(g, content_layer_name, content_features):
    content_loss = np.float32(0.0)
    with tf.Session(graph=g) as sess, g.device('/cpu:0'):
        content_loss = tf.nn.l2_loss((g.get_tensor_by_name(content_layer_name) -
                                      content_features) /
                                      content_features.size)
    return content_loss

def get_style_loss(g, style_layers, style_features):
    style_loss = np.float32(0.0)
    with tf.Session(graph=g) as sess, g.device('/cpu:0'):
        for style_layer_i, style_gram_i in zip(style_layers, style_features):
            layer_i = g.get_tensor_by_name(style_layer_i)
            print(layer_i.shape)
            layer_shape = layer_i.get_shape().as_list()
            layer_size = layer_shape[1] * layer_shape[2] * layer_shape[3]
            layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
            gram_matrix = tf.matmul(tf.transpose(layer_flat), layer_flat) / layer_size
            style_loss = tf.add(style_loss, tf.nn.l2_loss(gram_matrix - style_gram_i)/
                                np.float32(style_gram_i.size))
    return style_loss
    
def covert_gram_matrix(raw_activations):
    conved_activs = []
    for activation_i in raw_activations:
        s_i = np.reshape(activation_i, [-1, activation_i.shape[-1]])
        gram_matrix = np.matmul(s_i.T, s_i) /s_i.size
        conved_activs.append(gram_matrix.astype(np.float32))
    return conved_activs

def get_dropout_layer_names(g, names):
    d_names = [name_i for name_i in names if 'dropout'in name_i and name_i.endswith("random_uniform")]
    print(d_names)
    d_shapes = []

    dropout_dict = {}
    for idx, name in enumerate(d_names):
        print(idx, name)
        d_shapes.append(g.get_tensor_by_name(name + ':0').shape.as_list()[1])
    print(d_shapes)
    return d_names, d_shapes

def one_app():
    """

    """
    net, names, g = load_vgg16_model()
    x = g.get_tensor_by_name(names[0] + ':0')
    softmax = g.get_tensor_by_name(names[-2] + ':0')
    # print(x)
    # print(softmax.shape)

    # img = load_file("clinton.png")
    img = load_img_from_skimg()
    print(img.shape)
    img = vgg16.preprocess(img)
    print(img.shape)
    # plt.imshow(vgg16.deprocess(img))
    # plt.show()
    img_4d = img[np.newaxis]
    net_input = tf.Variable(img_4d)
    
    dropout_list = get_dropout_layer_names(g, names)
    # print(dropout_list[0], dropout_list[1][0])
    # res = eval_tensor(g, net, softmax, x, img_4d, dropout_list)
    # print(res)
    content_layer_name = 'vgg/conv4_2/conv4_2:0'
    content_features = get_tensor_eval(g, content_layer_name, x, img_4d, dropout_list)
    print(content_features.shape)
    
    # style 
    style_img = load_file("style.jpg")
    style_img = vgg16.preprocess(style_img)
    style_img_4d = style_img[np.newaxis]

    style_layers = ['vgg/conv1_1/conv1_1:0',
                    'vgg/conv2_1/conv2_1:0',
                    'vgg/conv3_1/conv3_1:0',
                    'vgg/conv4_1/conv4_1:0',
                    'vgg/conv5_1/conv5_1:0']
    style_activations = get_tensor_evals(g, style_layers, x, style_img_4d, dropout_list)
    style_features = covert_gram_matrix(style_activations)


    layer_i = g.get_tensor_by_name(style_layers[3])
    print(layer_i.shape)
    # print(style_features[0].shape)
    # content_loss = get_content_loss(g, content_layer_name, content_features)
    # style_loss = get_style_loss(g, style_layers, style_features)
    # style_acti_i = style_activations[0]
    # sh = style_acti_i.shape
    # print(sh, sh[0])

    # loss = 0.1 * content_loss + 5.0 * style_loss
    # optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    # with tf.Session(graph=g) as sess, g.device('/cup:0'):
    #     sess.run(tf.global_variables_initializer())
    #     n_iterations = 100
    #     og_img = net_input.eval()
    #     imgs = []
    #     for it_i in range(n_iterations):
    #         _, this_loss, synth = sess.run([optimizer, loss, net_input],
    #                                        feed_dict={
    #                                            'vgg/dropout_1/random_uniform:0':
    #                                            np.ones(g.get_tensor_by_name(
    #                                                'vgg/dropout_1/random_uniform:0').get_shape().as_list()),
    #                                            'vgg/dropout/random_uniform:0':
    #                                            np.ones(g.get_tensor_by_name(
    #                                                'vgg/dropout/random_uniform:0').get_shape().as_list())})
    #         print("%d: %f, (%f - %f)" %
    #               (it_i, this_loss, np.min(synth), np.max(synth)))
    #         if it_i % 5 == 0:
    #             imgs.append(np.clip(synth[0], 0, 1))
    #             fig, ax = plt.subplots(1, 3, figsize=(22, 5))
    #             ax[0].imshow(vgg16.deprocess(img))
    #             ax[0].set_title('content image')
    #             ax[1].imshow(vgg16.deprocess(style_img))
    #             ax[1].set_title('style image')
    #             ax[2].set_title('current synthesis')
    #             ax[2].imshow(vgg16.deprocess(synth[0]))
    #             plt.show()
    #             # fig.canvas.draw()
    #     gif.build_gif(imgs, saveto='stylenet-bosch.gif')

if __name__ == "__main__":
    # load_inception_model()
    one_app()


