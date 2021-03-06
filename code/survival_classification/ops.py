import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def conv2d(input_, output_dim, 
           k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv



def conv3d(input_, output_dim, k_d=3, k_h=3, k_w=3, d_d=1, d_h=1, d_w=1, pad_='SAME', stddev=0.02, name="conv3d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev));
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding=pad_);

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0));
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape());

        return conv;

def deconv3d(input_, output_shape, k_d=3, k_h=3, k_w=3, d_d=2, d_h=2, d_w=2, stddev=0.02, name="deconv3d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev));
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, strides=[1, d_d, d_h, d_w, 1]);

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0));
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape());

        if with_w:
            return deconv, w, biases;
        else:
            return deconv;

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x);

"""
def fc(input_, num_output_, name="fc", stddev=0.02):
    print "num output: ", num_output_;
    with tf.variable_scope(name):
        return tf.contrib.layers.fully_connected(inputs=input_, num_outputs=num_output_, activation_fn=tf.nn.sigmoid, \
                weights_initializer=tf.random_normal_initializer(stddev=stddev), \
                biases_initializer=tf.constant_initializer(0.0));
"""
def fc3d(input_, num_output_, name="fc", stddev=0.02):
    with tf.variable_scope(name):
        k_d = input_.get_shape()[1];
        k_h = input_.get_shape()[2];
        k_w = input_.get_shape()[3];
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], 1], initializer=tf.truncated_normal_initializer(stddev=stddev));
        conv = tf.nn.conv3d(input_, w, strides=[1, 1, 1, 1, 1], padding='VALID');
        biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0));
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape());

        return conv;

def fc2d(input_, num_output_, name="fc", stddev=0.02):
    with tf.variable_scope(name):
        k_h = input_.get_shape()[1];
        k_w = input_.get_shape()[2];
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], 1], initializer=tf.truncated_normal_initializer(stddev=stddev));
        conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID');
        biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0));
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape());

        return conv;
