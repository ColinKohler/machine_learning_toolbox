#!/usr/bin/python

import sys, os
sys.path.append('/home/colin/workspace/machine_learning_toolbox/')
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet_tf.alexnet import AlexNet
from utils.rigor_percept_importer import RigorPerceptImporter

dir_path = '/home/colin/workspace/machine_learning_toolbox/'

# Percepts
train_file = dir_path + 'metadata/trash_bag_train.json'
val_file = dir_path + 'metadata/trash_bag_validate.json'

# Learing params
lr = 0.0001
num_epochs = 2
batch_size = 128

# Network params
dropout_prob = 0.5
num_output = 2
train_layers = ['fc6', 'fc7', 'fc8']
iou_threshold = 0.5

# TensorBoard params
display_step = 1

filewriter_path = dir_path + 'models/trash_bags'
checkpoint_path = dir_path + 'models/'

def main():
    # TensorFlow stuff
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_output])
    keep_prob = tf.placeholder(tf.float32)

    # Init model and connect output to last layer
    model = AlexNet(x, keep_prob, num_output, train_layers)
    output = model.fc8
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # Setup loss function
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

    # Train Op
    with tf.name_scope('train'):
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Evaluation op
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Tensorboard stuff
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    for var in var_list:
        tf.summary.histogram(var.name, var)

    tf.summary.scalar('l2_loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    # Initialize thigns
    writer = tf.summary.FileWriter(filewriter_path)
    saver = tf.train.Saver()

    one_hot_encoding = {'no.trash.bag' : 0, 'trash.bag' : 1}
    train_importer = RigorPerceptImporter(train_file, 2, batch_size, [0.0, 0.0, 0.0], one_hot_encoding)
    val_importer = RigorPerceptImporter(val_file, 2, batch_size, [0.0, 0.0, 0.0], one_hot_encoding)
    train_batches_per_epoch = np.floor(train_importer.num_percepts / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(val_importer.num_percepts / batch_size).astype(np.int16)

    # Start TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        model.loadInitialWeights(sess)

        print '{} Start training...'.format(datetime.now())
        print '{} Open Tensorboard at --logdir {}'.format(datetime.now(), filewriter_path)

        for epoch in range(num_epochs):
            print '{} Epoch number: {}'.format(datetime.now(), epoch+1)

            step = 1
            # Train
            while step < train_batches_per_epoch:
                batch_xs, batch_ys = train_importer.getBatch()
                sess.run(train_op, feed_dict={x : batch_xs, y : batch_ys, keep_prob : dropout_prob})

                if step%display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x : batch_xs, y : batch_ys, keep_prob : 1.0})
                    writer.add_summary(s, epoch*train_batches_per_epoch + step)

                step += 1

            # Validate
            print '{} Start validation...'.format(datetime.now())

            test_acc = 0.0
            test_count = 0
            for _ in range(val_batches_per_epoch):
                tbatch_xs, tbatch_ys = val_importer.getBatch()
                acc = sess.run(accuracy, feed_dict={x : tbatch_xs, y : tbatch_ys, keep_prob : 1.0})
                test_acc += acc
                test_count += 1

            test_acc /= test_count
            print '{} Validation Accuracy = {:.4f}'.format(datetime.now(), test_acc)

            # Reset importer to all percepts
            train_importer.resetPointer()
            val_importer.resetPointer()

            # Save model
            checkpoint_name = os.path.join(checkpoint_path, '_{}.ckpt'.format(epoch))
            save_path = saver.save(sess, checkpoint_name)
            print '{} Saving checkpoint at {}'.format(datetime.now(), checkpoint_name)

if __name__ == '__main__':
    main()
