from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import math


def get_num_params():
    from functools import reduce
    from operator import mul
    num_params = 0
    print('All trainable variable: ')
    for variable in tf.trainable_variables():
        print('-- ', variable.name)
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


def compute_error(real,fake):
    return tf.reduce_mean(tf.abs(real-fake), reduction_indices=[0,1,2,3])


class Generator(object):
    def __init__(self, nin, nout_ch, first_layer_ch, bottleneck_sp=8, bottleneck_ch=256, res_block_num=4):
        self.nin = nin  # assuming nin is 'NHWC' format
        self.nout_ch = nout_ch
        self.first_layer_ch = first_layer_ch
        self.bottleneck_sp = bottleneck_sp
        self.bottleneck_ch = bottleneck_ch
        self.res_block_num = res_block_num

        self.nin_sp = nin.get_shape().as_list()[1]
        self.nin_ch = nin.get_shape().as_list()[3]
        self.encoder_layer_num = int(math.log(self.nin_sp/bottleneck_sp, 2)) + 1
        self.decoder_layer_num = self.encoder_layer_num
        self.layers = []
        self._build_net()

    def get_network_output(self):
        return self.layers[-1]

    def _build_net(self):
        print('Constructing generator with resolution of %dx%d' % (self.nin_sp,self.nin_sp))
        self.layers = []

        with tf.variable_scope('encoder_in'):
            net = slim.conv2d(self.nin, self.first_layer_ch, [1,1], stride=1,
                              padding='SAME',
                              weights_initializer=initializers.xavier_initializer_conv2d(),
                              weights_regularizer=None,
                              rate=1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu,
                              scope='conv0')
            self.layers.append(net)
            print('-- Layer %d: ' % len(self.layers), 'encoder_in ', self.layers[-1].get_shape().as_list())

        for i in range(1, self.encoder_layer_num, 1):
            sp = self.layers[-1].get_shape().as_list()[-2]
            with tf.variable_scope('encoder_%dx%d' % (sp, sp)):
                net = slim.conv2d(self.layers[-1], min(self.first_layer_ch*(2**i), self.bottleneck_ch), [4,4],
                                  stride=2, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer_conv2d(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu,
                                  scope='conv0')
                self.layers.append(net)
                print('-- Layer %d: ' % len(self.layers), 'encoder_%dx%d ' % (sp, sp), self.layers[-1].get_shape().as_list())

        for i in range(self.res_block_num):
            with tf.variable_scope('residual_block_%d' % i):
                net = slim.conv2d(self.layers[-1], self.bottleneck_ch, [3,3], stride=1, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer_conv2d(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=None, activation_fn=tf.nn.leaky_relu,
                                  scope='conv0')
                net = tf.add(net, self.layers[-1])
                self.layers.append(net)
                print('-- Layer %d: ' % len(self.layers), 'residual_block_%d ' % i, self.layers[-1].get_shape().as_list())

        for i in range(self.decoder_layer_num-1, 0, -1):
            sp = self.layers[-1].get_shape().as_list()[-2]
            with tf.variable_scope('decoder_%dx%d' % (sp*2, sp*2)):
                net = tf.image.resize_bilinear(self.layers[-1], (sp*2, sp*2), align_corners=True)
                net = slim.conv2d(net, min(self.first_layer_ch*(2**i), self.bottleneck_ch), [3,3],
                                  stride=1, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer_conv2d(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu,
                                  scope='conv0')
                net = tf.concat([net, self.layers[i-1], tf.image.resize_area(self.nin, (sp*2,sp*2), align_corners=False)], axis=3)
                net = slim.conv2d(net, min(self.first_layer_ch*(2**i), self.bottleneck_ch), [3,3],
                                  stride=1, padding='SAME',
                                  weights_initializer=initializers.xavier_initializer_conv2d(),
                                  weights_regularizer=None,
                                  rate=1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu,
                                  scope='conv1')
                self.layers.append(net)
                print('-- Layer %d: ' % len(self.layers), 'decoder_%dx%d ' % (sp*2, sp*2), self.layers[-1].get_shape().as_list())

        with tf.variable_scope('decoder_out'):
            net = slim.conv2d(self.layers[-1], self.nout_ch, [1,1], stride=1, padding='SAME',
                              weights_initializer=initializers.xavier_initializer_conv2d(),
                              rate=1, activation_fn=tf.nn.sigmoid, scope='conv0')
            self.layers.append(net)
            print('-- Layer %d: ' % len(self.layers), 'decoder_out ', self.layers[-1].get_shape().as_list())


class Discriminator(object):
    def __init__(self, nin, cond, first_layer_ch, disc_patch_res=32):
        self.nin = nin
        self.cond = cond
        self.first_layer_ch = first_layer_ch
        self.nin_sp = nin.get_shape().as_list()[1]
        self.nin_ch = nin.get_shape().as_list()[3]
        self.cond_sp = cond.get_shape().as_list()[1]
        self.cond_ch = cond.get_shape().as_list()[3]
        # self.first_layer_ch = self.nin_ch+self.cond_ch
        self.disc_patch_res = disc_patch_res
        self.encoder_layer_num = int(math.log(self.nin_sp / disc_patch_res, 2))+1
        self.layers = []

        self._build_net()

    def get_network_output(self):
        return self.layers[-1]

    def _build_net(self):
        print('Constructing discriminator with resolution of %dx%d' % (self.nin_sp,self.nin_sp))
        self.layers = []
        nin = tf.concat([self.nin, self.cond], axis=3)
        with tf.variable_scope('encoder_in'):
            net = slim.conv2d(nin, self.first_layer_ch, [1,1], stride=1,
                              padding='SAME',
                              weights_initializer=tf.random_normal_initializer(0, 0.02),
                              weights_regularizer=slim.l2_regularizer(0.0001),
                              rate=1, normalizer_fn=None, activation_fn=tf.nn.leaky_relu,
                              scope='conv0')
            self.layers.append(net)
            print('-- Layer %d: ' % len(self.layers), 'encoder_in ', self.layers[-1].get_shape().as_list())

        for i in range(1, self.encoder_layer_num, 1):
            sp = self.layers[-1].get_shape().as_list()[-2]
            with tf.variable_scope('encoder_%dx%d' % (sp, sp)):
                net = slim.conv2d(self.layers[-1], self.first_layer_ch*(2**i), [4,4],
                                  stride=2, padding='SAME',
                                  weights_initializer=tf.random_normal_initializer(0, 0.02),
                                  weights_regularizer=slim.l2_regularizer(0.0001),
                                  rate=1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu,
                                  scope='conv0')
                self.layers.append(net)
                print('-- Layer %d: ' % len(self.layers), 'encoder_%dx%d ' % (sp, sp), self.layers[-1].get_shape().as_list())

        with tf.variable_scope('encoder_out'):
            net = slim.conv2d(self.layers[-1], 1, [1,1], stride=1, padding='SAME',
                              weights_initializer=tf.random_normal_initializer(0, 0.02),
                              weights_regularizer=slim.l2_regularizer(0.0001),
                              rate=1,  normalizer_fn=None, activation_fn=tf.nn.sigmoid,
                              scope='conv0')
            self.layers.append(net)
            print('-- Layer %d: ' % len(self.layers), 'encoder_out ', self.layers[-1].get_shape().as_list())


