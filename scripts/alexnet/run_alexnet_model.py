#!/usr/bin/python

import tensorflow as tf
import argparse
import cv2
import numpy as np
from alexnet_tf.alexnet import Alexnet

def test(args):
    img = cv2.imread(args.image_path)
    img = cv2.resize(img, (227, 227))
    img = img.astype(np.float32)

    one_hot_encoding = loadClassEncoding(args.class_encoding_path)
    num_classes = len(one_hot_encoding)

    # TensorFlow stuff
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    # Init model and connect output to last layer
    model = Alexnet(x, keep_prob, num_classes, train=False)
    output = model.fc8
    pred = tf.argmax(output, 0)
    saver = tf.train.Saver()

    # Start TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.model_path)

        best = sess.run(output, feed_dict={x : img.reshape([1, 227, 227, 3]), keep_prob : 1.0})
        print best

        w = dict()
        for key in model.weights.keys():
            w[key] = sess.run(model.weights[key])
    np.save('trash_weights.npy', w)

# Load the class encoding
def loadClassEncoding(path):
    one_hot_encoding = dict()
    with open(path, 'r') as f:
        for line in f:
            cls, num = line.split(' ', 1)
            one_hot_encoding[cls] = int(num)
    return one_hot_encoding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to model to load')
    parser.add_argument('class_encoding_path', help='Path to the metadata file for the onehot encoding')
    parser.add_argument('image_path', type=str, help='Path to image to test model on')
    args = parser.parse_args()

    test(args)
