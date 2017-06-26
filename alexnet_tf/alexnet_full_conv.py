import sys
import tensorflow as tf
import numpy as np


class FullConvAlexnet(object):
    def __init__(self, x, output_num):
        self.x = x
        self.output_num = output_num

        self.createModel()

    def createModel(self):
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

        conv6 = self.conv(pool5, 6, 6, 4096, 1, 1, name='fc6-conv', padding='VALID', reuse=False)
        conv7 = self.conv(conv6, 1, 1, 4096, 1, 1, name='fc7-conv', reuse=False)

        self.conv8 = self.conv(conv7, 1, 1, self.output_num, 1, 1, name='fc8-conv', reuse=False)

    # Load weights from bvlc_alexnet.npy (Caffe weights)
    def loadInitialWeights(self, session, path):
        weights_dict = np.load(path).item()
        for op_name in weights_dict:
            tmp_name1 = op_name.split('_', 1)[0]
            tmp_name2 = tmp_name1.split('-', 1)[0]
            if 'fc' in tmp_name1:
                tmp_name1 = tmp_name1 + '-conv'
            with tf.variable_scope(tmp_name1, reuse=True):
                data = weights_dict[op_name]
                if len(data.shape) == 1:
                    var = tf.get_variable('biases', trainable=False)
                    session.run(var.assign(data))
                else:
                    var = tf.get_variable('weights', trainable=False)
                    if tmp_name2 == 'fc6':
                        session.run(var.assign(data.reshape([6,6,256,4096])))
                    elif tmp_name2 in ['fc7', 'fc8']:
                        session.run(var.assign(data.reshape([1,1]+list(data.shape))))
                    else:
                        session.run(var.assign(data))

    # Create Convolutional layer, to split computation use groups > 1
    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1, reuse=True):
        input_channels = int(x.get_shape()[-1])
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

        with tf.variable_scope(name, reuse=reuse) as scope:
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

            return relu

    # Create Fully-connected Layer
    def fc(self, x, num_in, num_out, name, relu=True, reuse=True):
        with tf.variable_scope(name, reuse=reuse) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', shape=[num_out], trainable=True)

            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
            relu = tf.nn.relu(act) if relu else act
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
