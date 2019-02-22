import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
import os, sys

def save_model():
    # prepare feed input, i.e. feed_dict and placeholder
    w1 = tf.placeholder('float', name='w1')
    w2 = tf.placeholder('float', name='w2')
    b1 = tf.Variable(2.0, name='b1')
    feed_dict = {w1:4, w2:8}

    # Ops
    op_add = tf.add(w1, w2)
    op_mul = tf.multiply(op_add, b1, name="op_to_restore")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # create saver object to save all variables
    saver = tf.train.Saver()

    # run the operation by feeding input
    print(sess.run(op_mul, feed_dict))

    # now save the graph
    saver.save(sess, './save_test/my_test_model', global_step=1000)

def restore_model():
    sess = tf.Session()
    # first, load meta graph and restore weights
    saver = tf.train.import_meta_graph('./save_test/my_test_model-1000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./save_test/'))

    print(sess.run('b1:0'))

    graph = tf.get_default_graph()

    names = [op.name for op in graph.get_operations()]
    print(names)

    w1 = graph.get_tensor_by_name('w1:0')
    w2 = graph.get_tensor_by_name('w2:0')
    feed_dict = {w1: 13, w2: 17}
    op_from_restore = graph.get_tensor_by_name('op_to_restore:0')

    print(sess.run(op_from_restore, feed_dict))

def save_pb():
    output_node_names = ['w1', 'w2', 'b1', 'op_to_restore']
    saver = tf.train.import_meta_graph('./save_test/my_test_model-1000.meta')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('./save_test/'))
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names)

    output_graph = './save_test/save_test.pb'
    with tf.gfile.GFile(output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()

def load_pb():
    pb_path = './save_test/save_test.pb'
    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as g:
        # tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="")
        retval = tf.import_graph_def(graph_def, input_map=None,
                            return_elements=['w1', 'w2', 'op_to_restore'], name="")
        print(retval)
    # tf.import_graph_def(graph_def)
    # g = tf.get_default_graph()
    names = [op.name for op in g.get_operations()]
    for i, name in enumerate(names):
        print(i, name)

    w1 = g.get_tensor_by_name("w1:0")
    w2 = g.get_tensor_by_name('w2:0')
    op_from_restore = g.get_tensor_by_name('op_to_restore:0')
    feed_dict = {w1: 2, w2: 3}

    with tf.Session(graph=g) as sess:
        # sess.run(tf.global_variables_initializer())
        print(sess.run(op_from_restore, feed_dict))


def save_variables():
    #create some variables.
    v1 = tf.get_variable('v1', shape=[3], initializer=tf.zeros_initializer)
    v2 = tf.get_variable('v2', shape=[5], initializer=tf.zeros_initializer)

    inc_v1 = v1.assign(v1 + 1)
    dec_v2 = v2.assign(v2 - 1)

    # op to initialize the variables
    init_op = tf.global_variables_initializer()

    # create saver to save and restore all the variables
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(inc_v1)
        sess.run(dec_v2)

        save_path = saver.save(sess, './save_variables/model.ckpt')
        print('model saved in path: %s' % save_path)

def restore_variables():
    tf.reset_default_graph()
    v1 = tf.get_variable("v1", shape=[3])
    v2 = tf.get_variable("v2", shape=[5])

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, './save_variables/model.ckpt')
        print('model restored.')
        print("v1: %s" % v1.eval())
        print("v2: %s" % v2.eval())

def save_variables_part():
    #create some variables.
    v1 = tf.get_variable('v1', shape=[3], initializer=tf.zeros_initializer)
    v2 = tf.get_variable('v2', shape=[5], initializer=tf.zeros_initializer)

    inc_v1 = v1.assign(v1 + 1)
    dec_v2 = v2.assign(v2 - 1)

    # op to initialize the variables
    init_op = tf.global_variables_initializer()

    # create saver to save and restore all the variables
    saver = tf.train.Saver({'v2': v2})

    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(inc_v1)
        sess.run(dec_v2)

        save_path = saver.save(sess, './save_variables/model_part.ckpt')
        print('model saved in path: %s' % save_path)

def restore_variables_part():
    tf.reset_default_graph()
    v1 = tf.get_variable("v1", shape=[3])
    v2 = tf.get_variable("v2", shape=[5])

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, './save_variables/model_part.ckpt')
        print('model restored.')
        print("v1: %s" % v1.eval())
        print("v2: %s" % v2.eval())


if __name__ == '__main__':
    # save_variables()
    # restore_variables()
    # save_variables_part()
    # save_model()
    # save_pb()
    load_pb()
    # restore_model()
