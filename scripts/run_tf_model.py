import tensorflow as tf
import argparse
import cv2
import numpy as np
from alexnet_tf.alexnet import AlexNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to model to load')
    parser.add_argument('image_path', type=str, help='Path to image to test model on')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img = cv2.resize(img, (227, 227))
    img = img.astype(np.float32)

    num_output = 2
    train_layers = []

    # TensorFlow stuff
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    # Init model and connect output to last layer
    model = AlexNet(x, keep_prob, num_output, train_layers)
    output = model.fc8
    pred = tf.argmax(output, 1)
    saver = tf.train.Saver()

    # Start TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.model_path)

        best = sess.run(pred, feed_dict={x : img.reshape([1, 227, 227, 3]), keep_prob : 1.0})
        print best

if __name__ == '__main__':
    main()
