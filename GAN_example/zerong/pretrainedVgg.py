from __future__ import division, print_function, absolute_import
import numpy as np
import os, time, scipy.io
import tensorflow as tf 


def build_pretrained_vgg_layer_conv(nin, nwb, name=None):
    return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1,1,1,1], padding='SAME', name=name) + nwb[1])


def build_pretrained_vgg_layer_pool(nin):
    return tf.nn.avg_pool(nin, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def get_pretrained_vgg_wb(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def build_pretrained_vgg(nin, model_dir, reuse = False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    vgg_rawnet = scipy.io.loadmat(model_dir)
    vgg_layers = vgg_rawnet['layers'][0]
    
    net = {}
    net['input'] = nin - np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
    net['conv1_1'] = build_pretrained_vgg_layer_conv(net['input'], get_pretrained_vgg_wb(vgg_layers, 0), name='vgg_conv1_1')
    net['conv1_2'] = build_pretrained_vgg_layer_conv(net['conv1_1'], get_pretrained_vgg_wb(vgg_layers, 2), name='gbb_conv1_2')
    net['pool1'] = build_pretrained_vgg_layer_pool(net['conv1_2'])

    net['conv2_1'] = build_pretrained_vgg_layer_conv(net['pool1'], get_pretrained_vgg_wb(vgg_layers, 5), name='vgg_conv2_1')
    net['conv2_2'] = build_pretrained_vgg_layer_conv(net['conv2_1'], get_pretrained_vgg_wb(vgg_layers, 7), name='vgg_conv2_2')
    net['pool2'] = build_pretrained_vgg_layer_pool(net['conv2_2'])

    net['conv3_1'] = build_pretrained_vgg_layer_conv(net['pool2'], get_pretrained_vgg_wb(vgg_layers, 10), name='vgg_conv3_1')
    net['conv3_2'] = build_pretrained_vgg_layer_conv(net['conv3_1'], get_pretrained_vgg_wb(vgg_layers, 12), name='vgg_conv3_2')
    net['conv3_3'] = build_pretrained_vgg_layer_conv(net['conv3_2'], get_pretrained_vgg_wb(vgg_layers, 14), name='vgg_conv3_3')
    net['conv3_4'] = build_pretrained_vgg_layer_conv(net['conv3_3'], get_pretrained_vgg_wb(vgg_layers, 16), name='vgg_conv3_4')
    net['pool3'] = build_pretrained_vgg_layer_pool(net['conv3_4'])
    
    net['conv4_1'] = build_pretrained_vgg_layer_conv(net['pool3'], get_pretrained_vgg_wb(vgg_layers, 19), name='vgg_conv4_1')
    net['conv4_2'] = build_pretrained_vgg_layer_conv(net['conv4_1'], get_pretrained_vgg_wb(vgg_layers, 21), name='vgg_conv4_2')
    net['conv4_3'] = build_pretrained_vgg_layer_conv(net['conv4_2'], get_pretrained_vgg_wb(vgg_layers, 23), name='vgg_conv4_3')
    net['conv4_4'] = build_pretrained_vgg_layer_conv(net['conv4_3'], get_pretrained_vgg_wb(vgg_layers, 25), name='vgg_conv4_4')
    net['pool4'] = build_pretrained_vgg_layer_pool(net['conv4_4'])
    
    net['conv5_1'] = build_pretrained_vgg_layer_conv(net['pool4'], get_pretrained_vgg_wb(vgg_layers, 28), name='vgg_conv5_1')
    net['conv5_2'] = build_pretrained_vgg_layer_conv(net['conv5_1'], get_pretrained_vgg_wb(vgg_layers, 30), name='vgg_conv5_2')
    net['conv5_3'] = build_pretrained_vgg_layer_conv(net['conv5_2'], get_pretrained_vgg_wb(vgg_layers, 32), name='vgg_conv5_3')
    net['conv5_4'] = build_pretrained_vgg_layer_conv(net['conv5_3'], get_pretrained_vgg_wb(vgg_layers, 34), name='vgg_conv5_4')
    net['pool5'] = build_pretrained_vgg_layer_pool(net['conv5_4'])

    return net
