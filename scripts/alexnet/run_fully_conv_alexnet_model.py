#!/usr/bin/python

import tensorflow as tf
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tensorflow_networks.alexnet import Alexnet
import utils.tf_utils as tf_utils

def run(args):
    img_size = 579
    img = cv2.imread(args.image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)

    one_hot_encoding = tf_utils.loadClassEncoding(args.class_encoding_path)
    num_classes = len(one_hot_encoding)

    # TensorFlow stuff
    x = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

    # Init model and connect output to last layer
    model = Alexnet(x, num_classes, full_conv=True)
    saver = tf.train.Saver()

    # Start TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.model_path)

        cells = (img_size - 227) / 32 + 1
        best = sess.run(model.output, feed_dict={x : img.reshape([1, img_size, img_size, 3]), model.keep_prob : 1.0})
        best_cell = np.unravel_index(best[0,:,:,1].argmax(), [cells,cells])
        min_point = (best_cell[1] * 32, best_cell[0] * 32)
        print best_cell

        fig, ax = plt.subplots(1)
        ax.imshow(rgb_img)
        sx = width / float(img_size)
        sy = height / float(img_size)
        p = (int(min_point[0]*sy), int(min_point[1]*sx))
        lx = int(227 * sx)
        ly = int(227 * sy)
        print p
        print lx
        print ly
        rect = patches.Rectangle(p, ly, lx, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()

        #plt.subplot(1,2,1)
        #plt.imshow(rgb_img)
        #plt.subplot(1,2,2)
        #plt.imshow(best[0,:,:,1])
        #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to model to load')
    parser.add_argument('class_encoding_path', help='Path to the metadata file for the onehot encoding')
    parser.add_argument('image_path', type=str, help='Path to image to test model on')
    args = parser.parse_args()

    run(args)
