import sys, os, flask
from flask import render_template, request, url_for, redirect, send_from_directory, jsonify
from werkzeug import secure_filename
import tensorflow as tf
import numpy as np

app = flask.Flask(__name__)

def load_graph():
    model_path = './static/save_test.pb'
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def, input_map=None,
                           return_elements=None, name="")
    return g

@app.route('/')
def index():
    return render_template('load_pb.html')

@app.route('/data_input', methods = ['POST', 'GET'])
def data_input():
    if request.method == 'POST':
        usr_w1 = request.form['usr_w1']
        usr_w2 = request.form['usr_w2']

        cur_graph = app.graph
        w1 = cur_graph.get_tensor_by_name("w1:0")
        w2 = cur_graph.get_tensor_by_name("w2:0")
        op_to_restore = cur_graph.get_tensor_by_name("op_to_restore:0")
        feed_dict = {w1 : usr_w1, w2 : usr_w2}

        with tf.Session(graph=cur_graph) as sess:
            result = sess.run(op_to_restore, feed_dict)
            out = {"value" : str(result)}
            # return jsonify(out)
        return jsonify(out)
        # return "working" + str(result)

app.graph = load_graph()

if __name__ == '__main__':
    app.run(host='0:0:0:0', port=int(5000), debug=True, use_reloader=False)

