import sys, os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet

dir_path = '/home/colin/workspace/machine_learning_tools/'

# Percepts
train_file = dir_path + 'metadata/trash_bag_train.txt'
val_file = dir_path + 'metadata/trash_bag_validate.txt'

# Learing params
lr = 0.0001
num_epochs = 2
batch_size = 128

# Network params
dropout_prob = 0.5
num_outputs = [2, 4]
skip_layers = ['fc6', 'fc7', 'fc8']
train_layers_1 = ['fc6_1', 'fc7_1', 'fc8_1']
train_layers_2 = ['fc6_2', 'fc7_2', 'fc8_2']
iou_threshold = 0.5

# TensorBoard params
display_step = 1

filewriter_path = dir_path + 'models/trash_bags'
checkpoint_path = dir_path + 'models/'

# TensorFlow stuff
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y1 = tf.placeholder(tf.float32, [None, num_outputs[0]])
y2 = tf.placeholder(tf.float32, [None, num_outputs[1]])
keep_prob = tf.placeholder(tf.float32)

# Init model and connect output to last layer
model = AlexNet(x, keep_prob, num_outputs, skip_layers)
outputs = [model.fc8_1, model.fc8_2]
var_list_1 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers_1]
var_list_2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers_2]

# Setup loss function
with tf.name_scope("loss"):
    loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs[0], labels=y1)
    loss_2 = tf.reduce_mean(tf.nn.l2_loss(outputs[1] - y2))

# Train Op
with tf.name_scope('train'):
    gradients_1 = tf.gradients(loss_1, var_list_1)
    gradients_1 = list(zip(gradients, var_list_1))
    gradients_2 = tf.gradients(loss_2, var_list_2)
    gradients_2 = list(zip(gradients, var_list_2))

    optimizer_1 = tf.train.GradientDescentOptimizer(lr)
    optimizer_2 = tf.train.GradientDescentOptimizer(lr)
    trainop_1 = optimizer_1.apply_gradients(grads_and_vars=gradients_1)
    trainop_2 = optimizer_2.apply_gradients(grads_and_vars=gradients_2)

# Evaluation op
with tf.name_scope('accuracy'):
    correct_pred_1 = tf.equal(tf.argmax(outputs[0], 1), tf.argmax(y1, 1))
    accuracy_1 = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    x1 = tf.maximum(x[0], y[0])
    y1 = tf.maximum(x[1], y[1])
    x2 = tf.minimum(x[2], y[2])
    y2 = tf.minimum(x[3], y[3])

    inter_area = tf.multiply(tf.add(tf.subtract(x[, x1), 1.0),
                             tf.add(tf.subtract(y2, y1), 1.0))
    box1_area = tf.multiply(tf.add(tf.subtract(x[2], x[0]), 1.0),
                            tf.add(tf.subtract(x[3], x[1]), 1.0))
    box2_area = tf.multiply(tf.add(tf.subtract(y[2], y[0]), 1.0),
                            tf.add(tf.subtract(y[3], y[1]), 1.0))
    outer_area = tf.subtract(tf.add(box1_area, box2_area), inter_area)
    iou = tf.div(inter_area, outer_area)

    correct_pred_2 = tf.greater_equal(iou, iou_threshold)
    accuracy_2 = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Tensorboard stuff
for gradient, var in gradients_1:
    tf.summary.histogram(var.name + '/gradient_1', gradient)
for gradient, var in gradients_2:
    tf.summary.histogram(var.name + '/gradient_2', gradient)

for var in var_list_1:
    tf.summary.histogram(var.name, var)
for var in var_list_2:
    tf.summary.histogram(var.name, var)

tf.summary.scalar('l2_loss_1', loss_1)
tf.summary.scalar('l2_loss_2', loss_2)
tf.summary.scalar('accuracy_1', accuracy_1)
tf.summary.scalar('accuracy_2', accuracy_2)
merged_summary = tf.summary.merge_all()

# Initialize thigns
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()
train_batches_per_epoch = np.floor().astype(np.int16)
val_batches_per_epoch = np.floor().astype(np.int16)

# Start TF session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    model.load_initial_weights(sess)

    print '{} Start training...'.format(datetime.now())
    print '{} Open Tensorboard at --logdir {}'.format(datetime.now(), filewriter_path)

    for epoch in range(num_epochs):
        print '{} Epoch number: {}'.format(datetime.now(), epoch+1)

        step = 1
        # Train
        while step < train_batches_per_epoch:
            batch_xs, batch_ys =
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
            tbatch_xs, tbatch_ys =
            acc = sess.run(accuracy, feed_dict={x : tbatch_xs, y : tbatch_ys, keep_prob : 1.0})
            test_acc += acc
            test_count += 1

        test_acc /= test_count
        print '{} Validation Accuracy = {:.4f}'.format(datetime.now(), test_acc)

        # Save model
        checkpoint_name = os.path.join(checkpoint_path, '_{}.ckpt'.format(epoch))
        save_path = saver.save(sess, checkpoint_name)
        print '{} Saving checkpoint at {}'.format(datetime.now(), checkpoint_name)
