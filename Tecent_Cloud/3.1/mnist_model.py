# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def deepnn(x):

    with tf.name_scope('reshpae'):
        x_image = tf.reshape(x, [-1,28,28,1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_Variable([5,5,1,32])
        b_conv1 = bias_Variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_Variable([5,5,32,64])
        b_conv2 = bias_Variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_Variable([7*7*64,1024])
        b_fc1 = bias_Variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_Variable([1024, 10])
        b_fc2 = bias_Variable([10])

        y_conv = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv, keep_prob

#卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

#池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")

#权重
def weight_Variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏置
def bias_Variable(shape):
    initial = tf.zeros(shape) + 0.1 #tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

