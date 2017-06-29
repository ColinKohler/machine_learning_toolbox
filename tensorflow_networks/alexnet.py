import sys
import tensorflow as tf
import numpy as np

WEIGHTS_PATH = '/home/colin/workspace/machine_learning_toolbox/tensorflow_networks/bvlc_alexnet.npy'

class Alexnet(object):
    def __init__(self, x, output_num, lr=None, y=None, skip_layer=list(), train=False, full_conv=False, reuse=False):
        self.x = x
        self.output_num = output_num
        self.keep_prob = tf.placeholder(tf.float32)
        self.y = y
        self.lr = lr

        self.train = train
        self.skip_layer = skip_layer
        self.reuse = reuse

        self.weights = dict()
        self.createModel(full_conv)

    def createModel(self, full_conv):
        conv1 = self.conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = self.max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = self.lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        conv2 = self.conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = self.max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = self.lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        conv3 = self.conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        conv4 = self.conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        conv5 = self.conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        if full_conv:
            conv6 = self.fc_conv(pool5, 6, 6, 4096, name='fc6')
            conv7 = self.fc_conv(conv6, 1, 1, 4096, name='fc7')
            self.conv8 = self.fc_conv(conv7, 1, 1, self.output_num, name='fc8')
            self.output = self.conv8
        else:
            pool5_shape = pool5.get_shape()
            fc6_size = int(pool5_shape[1]) * int(pool5_shape[2]) * int(pool5_shape[3])
            flattened = tf.reshape(pool5, [-1, fc6_size])
            fc6 = self.fc(flattened, fc6_size, 4096, name='fc6')
            dropout6 = self.dropout(fc6, self.keep_prob)
            fc7 = self.fc(dropout6, 4096, 4096, name='fc7')
            dropout7 = self.dropout(fc7, self.keep_prob)
            self.fc8 = self.fc(dropout7, 4096, self.output_num, relu=False, name='fc8')
            self.output = self.fc8

        if self.y is not None:
            # Evaluation op
            with tf.name_scope('accuracy'):
                correct_pred = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if self.train:
            # Setup loss
            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=self.y))

            # Train Op
            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = optimizer.minimize(self.loss)

            # Tensorboard stuff
            tf.summary.scalar('l2_loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.merged_summary = tf.summary.merge_all()

    # Load weights from bvlc_alexnet.npy (Caffe weights)
    def loadInitialWeights(self, session):
        weights_dict = np.load(WEIGHTS_PATH, encoding='bytes').item()
        for op_name in weights_dict:
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

    # Create Convolutional layer, to split computation use groups > 1
    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
        input_channels = int(x.get_shape()[-1])
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

        with tf.variable_scope(name, reuse=self.reuse) as scope:
            weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

            # Handle splitting of conv layers onto 2 GPUs
            if groups == 1:
                conv = convolve(x, weights)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
                output_groups = [convolve(i, k) for i,k in zip(input_groups,weight_groups)]

                conv = tf.concat(axis=3, values=output_groups)

            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            relu = tf.nn.relu(bias, name=scope.name)

            self.weights[name+'_w'] = weights
            self.weights[name+'_b'] = biases

            return relu

    # Converts a fully connected layer into a conv layer
    def fc_conv(self, x, filter_height, filter_width, num_filters, name):
        with tf.variable_scope(name, reuse=self.reuse) as scope:
            in_shape = x.get_shape()
            channels = int(in_shape[-1])
            weights = tf.get_variable('weights', shape=[filter_height*filter_width*channels, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
            reshape_weights = tf.reshape(weights, shape=[filter_height, filter_width, channels, num_filters])
            conv = tf.nn.conv2d(x, reshape_weights, strides=[1,1,1,1], padding='VALID')
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            relu = tf.nn.relu(bias, name=scope.name)

            self.weights[name+'_w'] = weights
            self.weights[name+'_b'] = biases

            return relu

    # Create Fully-connected Layer
    def fc(self, x, num_in, num_out, name, relu=True):
        with tf.variable_scope(name, reuse=self.reuse) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', shape=[num_out], trainable=True)

            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
            relu = tf.nn.relu(act) if relu else act

            self.weights[name+'_w'] = weights
            self.weights[name+'_b'] = biases

            return relu

    # Create max pooling layer
    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

    # Create local response normalization layer
    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

    # Create dropout layer
    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)
