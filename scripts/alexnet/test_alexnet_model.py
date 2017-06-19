#!/usr/bin/python

import tensorflow as tf
import argparse
import cv2
import numpy as np
from alexnet_tf.alexnet import Alexnet
from utils.rigor_percept_importer import RigorPerceptImporter

def testModel(args):
    batch = 128
    one_hot_encoding = loadClassEncoding(args.class_encoding_path)
    num_classes = len(one_hot_encoding)
    test_importer = RigorPerceptImporter(args.test_metadata_path, batch, [0.0, 0.0, 0.0], args.class_encoding_path)
    test_batches_per_epoch = np.floor(test_importer.num_percepts / batch).astype(np.int16)

    # TensorFlow stuff
    x = tf.placeholder(tf.float32, [batch, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # Init model and connect output to last layer
    model = Alexnet(x, keep_prob, num_classes, y=y, train=False)
    output = model.fc8
    pred = tf.argmax(output, 0)
    saver = tf.train.Saver()

    # Start TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.model_path)

        test_acc = 0.0
        test_count = 0
        for _ in range(test_batches_per_epoch):
            tbatch_xs, tbatch_ys = test_importer.getBatch()
            acc = sess.run(model.accuracy, feed_dict={x : tbatch_xs, y : tbatch_ys, keep_prob : 1.0})
            test_acc += acc
            test_count += 1

        test_acc /= test_count
        print 'Test Accuracy = {:.4f}'.format(test_acc)

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
    parser.add_argument('test_metadata_path', type=str, help='Path to metadata file for the training data')
    args = parser.parse_args()

    testModel(args)