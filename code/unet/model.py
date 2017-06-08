from __future__ import division
import os
import time
import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import scipy.io as sio
from math import floor

from ops import *
from utils import *


class CAE_3D(object):
    def __init__(self, sess):
        self.sess = sess;
        self.batch_size = 1;
        self.image_size = [64, 128, 128, 1];      # [depth, height, width, channel]

        self.training_dir_list = ['/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_60_tumor', '/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_61_180_tumor'];

        self.build_model();

    def build_model(self):
        self.input_image = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]], name='input_image');

        self.reconstructed = self.generator(self.input_image);

        self.reconstruction_loss = tf.reduce_mean(tf.abs(self.input_image - self.reconstructed));
        t_vars = tf.trainable_variables();
        self.g_vars = [var for var in t_vars if 'g_' in var.name];
        
    def train(self, args):
        optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.reconstruction_loss, var_list=self.g_vars);

        print "Begin loading data";
        self.load_training_data();
        print "Finish loading data";

        init_op = tf.global_variables_initializer();
        self.sess.run(init_op);

        counter = 0;
        loss_sum = 0;
        loss_count = 0;
        for epoch in xrange(args.epoch):
            batch_idxs = len(self.X_train) // self.batch_size;

            for idx in xrange(0, batch_idxs):
                batch_images = self.X_train[idx*self.batch_size:(idx+1)*self.batch_size];

                self.sess.run([optim], feed_dict={self.input_image: batch_images});

                loss_value = self.reconstruction_loss.eval({self.input_image: batch_images});
                loss_sum += loss_value;
                loss_count += 1;
                
                if np.mod(counter, self.X_train.shape[0]) == 5:
                    print("Loss %4.4f " % (loss_sum / loss_count));
                    loss_sum = 0;
                    loss_count = 0;

                counter += 1;


    def generator(self, image):
        with tf.variable_scope("generator") as scope:
            self.gf_dim = 64;
            self.output_size = 1;
            
            s_d = self.image_size[0];
            s_h = self.image_size[1];
            s_w = self.image_size[2];
            s_d_2, s_d_4, s_d_8, s_d_16, s_d_32, s_d_64, s_d_128 = int(s_d/2), int(s_d/4), int(s_d/8), int(s_d/16), int(s_d/32), int(s_d/64), int(s_d/128);
            s_h_2, s_h_4, s_h_8, s_h_16, s_h_32, s_h_64, s_h_128 = int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16), int(s_h/32), int(s_h/64), int(s_h/128);
            s_w_2, s_w_4, s_w_8, s_w_16, s_w_32, s_w_64, s_w_128 = int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16), int(s_w/32), int(s_w/64), int(s_w/128);

            # Encoding part
            e1 = conv3d(image, self.gf_dim, name='g_e1_conv');
            e2 = conv3d(lrelu(e1), self.gf_dim*2, name='g_e2_conv');
            e3 = conv3d(lrelu(e2), self.gf_dim*2, name='g_e3_conv');

            # Decoding part
            self.d1, self.d1_w, self.d1_b = deconv3d(tf.nn.relu(e3), [self.batch_size, s_d_4, s_h_4, s_w_4, self.gf_dim*2], name='g_d1', with_w=True);
            d1 = tf.nn.dropout(self.d1, 0.5);
            d1 = tf.concat([d1, e2], 4);

            self.d2, self.d2_w, self.d2_b = deconv3d(tf.nn.relu(d1), [self.batch_size, s_d_2, s_h_2, s_w_2, self.gf_dim], name='g_d2', with_w=True);
            d2 = tf.nn.dropout(self.d2, 0.5);
            d2 = tf.concat([d2, e1], 4);

            self.d3, self.d3_w, self.d3_b = deconv3d(tf.nn.relu(d2), [self.batch_size, s_d, s_h, s_w, 1], name='g_d3', with_w=True);

            return tf.nn.tanh(self.d3);


    def load_training_data(self):
        X_train = np.zeros(shape=(0, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        for training_folder in self.training_dir_list:
            X_tr = self.load_training_data_folder(training_folder);
            X_train = np.concatenate((X_train, X_tr));

        mu = np.mean(X_train[0::1, :, :, :].flatten());
        sigma = np.std(X_train[0::1, :, :, :].flatten());
        print "mu, sigma: ", mu, sigma;
        X_train = (X_train - mu) / sigma;

        self.X_train = X_train;
        print "X_train shape: ", self.X_train.shape;

    def load_training_data_folder(self, folder):
        X_train = np.zeros(shape=(400, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        
        idx = 0;
        for file_name in glob.glob(folder + '/*.mat'):
            loaded = sio.loadmat(file_name);
            img = loaded['norm_tumor'];
            img = np.transpose(img, (2, 0, 1));
            X_train[idx, :, :, :, 0] = img;
            idx += 1;

        X_train = X_train[:idx];

        return X_train;


    def test(self):
        init_op = tf.global_variables_initializer();
        self.sess.run(init_op);

        synthesis_input = np.random.rand(1, self.image_size[0], self.image_size[1], self.image_size[2], 1);
        samples = self.sess.run(self.reconstructed, feed_dict={self.input_image: synthesis_input});
        print "samples shape ", samples.shape;
        print samples[0, 1:3, 1:3, 1:3, 0];




