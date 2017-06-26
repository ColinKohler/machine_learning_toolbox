#!/usr/bin/python

import tensorflow as tf
import argparse
import cv2
import numpy as np
from alexnet_tf.alexnet_full_conv import FullConvAlexnet
import matplotlib.pyplot as plt

def run(args):
    img = cv2.imread(args.image_path)
    img = cv2.resize(img, (451, 451))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    one_hot_encoding = loadClassEncoding(args.class_encoding_path)
    num_classes = len(one_hot_encoding)

    # TensorFlow stuff
    x = tf.placeholder(tf.float32, [1, 451, 451, 3])
    keep_prob = tf.placeholder(tf.float32)

    # Init model and connect output to last layer
    model = FullConvAlexnet(x, num_classes)
    output = model.conv8

    # Start TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.loadInitialWeights(sess)

        best = sess.run(output, feed_dict={x : img.reshape([1, 451, 451, 3]), keep_prob : 1.0})
        print best[0].argmax(axis=2)
        print np.unravel_index(best[0,:,:,1].argmax(), [8,8])

        #plt.subplot(1,2,1)
        #plt.imshow(rgb_img)
        #plt.subplot(1,2,2)
        #plt.imshow(best[0,:,:,1])
        #plt.show()

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
    parser.add_argument('class_encoding_path', help='Path to the metadata file for the onehot encoding')
    parser.add_argument('image_path', type=str, help='Path to image to test model on')
    args = parser.parse_args()

    run(args)
