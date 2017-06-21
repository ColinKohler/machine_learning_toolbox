#!/usr/bin/python

import os
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
from alexnet_tf.alexnet import Alexnet
from utils.rigor_percept_importer import RigorPerceptImporter

def train(args):
    # Get training and validation data
    train_importer = RigorPerceptImporter(args.train_metadata_path, args.batch, [0.0, 0.0, 0.0], args.class_encoding_path)
    val_importer = RigorPerceptImporter(args.val_metadata_path, args.batch, [0.0, 0.0, 0.0], args.class_encoding_path)
    train_batches_per_epoch = np.floor(train_importer.num_percepts / args.batch).astype(np.int16)
    val_batches_per_epoch = np.floor(val_importer.num_percepts / args.batch).astype(np.int16)
    num_output = train_importer.num_classes

    # Init tensorflow model
    x = tf.placeholder(tf.float32, [args.batch, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_output])
    keep_prob = tf.placeholder(tf.float32)
    model = Alexnet(x, keep_prob, num_output, y=y, lr=args.lr, skip_layer=args.finetune_layers)
    output = model.fc8

    writer = tf.summary.FileWriter(args.tensorboard_path)
    saver = tf.train.Saver()

    # Start TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        if args.finetune_layers:
            model.loadInitialWeights(sess)

        print '{} Start training...'.format(datetime.now())
        print '{} Open Tensorboard at --logdir {}'.format(datetime.now(), args.tensorboard_path)

        for epoch in range(args.epochs):
            print '{} Epoch number: {}'.format(datetime.now(), epoch+1)

            step = 1
            # Train
            while step < train_batches_per_epoch:
                batch_xs, batch_ys = train_importer.getBatch()
                sess.run(model.train_op, feed_dict={x : batch_xs, y : batch_ys, keep_prob : args.dropout})

                if step%args.log_step == 0:
                    s = sess.run(model.merged_summary, feed_dict={x : batch_xs, y : batch_ys, keep_prob : 1.0})
                    writer.add_summary(s, epoch*train_batches_per_epoch + step)

                step += 1

            # Validate
            print '{} Start validation...'.format(datetime.now())

            test_acc = 0.0
            test_count = 0
            for _ in range(val_batches_per_epoch):
                tbatch_xs, tbatch_ys = val_importer.getBatch()
                acc = sess.run(model.accuracy, feed_dict={x : tbatch_xs, y : tbatch_ys, keep_prob : 1.0})
                test_acc += acc
                test_count += 1

            test_acc /= test_count
            print '{} Validation Accuracy = {:.4f}'.format(datetime.now(), test_acc)

            # Reset importer to all percepts
            train_importer.resetPointer()
            val_importer.resetPointer()

            # Save model
            checkpoint_name = os.path.join(args.checkpoint_path, '_{}.ckpt'.format(epoch))
            save_path = saver.save(sess, checkpoint_name)
            print '{} Saving checkpoint at {}'.format(datetime.now(), checkpoint_name)
    if args.save_weights:
        model.saveWeightsAsNpy('weights.npy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a classifier on a Alexnet model')
    parser.add_argument('class_encoding_path', help='Path to the metadata file for the onehot encoding')
    parser.add_argument('train_metadata_path', help='Path to the metadata file for the training data')
    parser.add_argument('val_metadata_path', help='Path to the metadata file for the validation data')
    parser.add_argument('checkpoint_path', help='Path to save the trained models at.')
    parser.add_argument('tensorboard_path', help='Path to log tensorboard info at.')

    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train over')
    parser.add_argument('--batch', default=128, type=int, help='Batch size')
    parser.add_argument('--finetune_layers', default=list(), nargs='+', help='Specify layers to finetune')
    parser.add_argument('--log_step', default=1, type=int, help='Log tensorbaord info every n steps')
    parser.add_argument('--save_weights', dest='save_weights', action='store_true')
    parser.set_defaults(save_weights=False)
    train(parser.parse_args())
