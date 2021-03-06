from __future__ import division
import os
import time
import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import scipy.io as sio
from math import floor
import sys
import scipy.misc

from ops import *
from utils import *


class CAE_3D(object):
    def __init__(self, sess, dataset_name='nsclc', checkpoint_dir=None):
        self.sess = sess;
        self.division_norm = 700;      # This number is used to divide to norm the output to 0 --> 1 range
        self.batch_size = 1;
        self.image_size = [64, 128, 128, 1];      # [depth, height, width, channel]
        self.dataset_name = dataset_name;
        self.checkpoint_dir = checkpoint_dir;

        self.training_dir_list = ['/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_61_180_tumor', \
                                    '/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_181_422_tumor'];
        self.val_dir_list = ['/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_60_tumor'];

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')


        self.build_model();

    def build_model(self):
        self.input_image = tf.placeholder(tf.float32, [self.batch_size, self.image_size[1], self.image_size[2], self.image_size[3]], name='input_image');
        self.raw_groundtruth = tf.placeholder(tf.float32, [self.batch_size], name='raw_output');
        self.norm_groundtruth = tf.sigmoid(self.raw_groundtruth / self.division_norm);

        self.pred_survival = self.generator(self.input_image);

        self.residual_loss = tf.reduce_mean(tf.abs(self.pred_survival - self.norm_groundtruth));
        t_vars = tf.trainable_variables();
        self.g_vars = [var for var in t_vars if 'g_' in var.name];
        print "No. of trained vars: ", len(self.g_vars);
        for g_v in self.g_vars:
            print g_v.name;

        self.saver = tf.train.Saver()

    def validate(self):
        pred_days = np.zeros(shape=(self.X_val.shape[0],), dtype=np.float32);
        gt_days = np.zeros(shape=(self.X_val.shape[0],), dtype=np.float32);
        batch_idxs = len(self.X_val) // self.batch_size;
        loss_sum = 0;
        loss_count = 0;
        for idx in xrange(0, batch_idxs):
            batch_images = self.X_val[idx*self.batch_size:(idx+1)*self.batch_size];
            batch_groundtruths = self.Y_val[idx*self.batch_size:(idx+1)*self.batch_size];
            loss_value, normgroundtruth, pred_sur = self.sess.run([self.residual_loss, self.norm_groundtruth, self.pred_survival], \
                                                feed_dict={self.input_image: batch_images, self.raw_groundtruth: batch_groundtruths});
            #print pred_sur.shape
            pred_days[loss_count] = pred_sur;
            gt_days[loss_count] = normgroundtruth;
            loss_sum += loss_value;
            loss_count += 1;

        #pred_days = np.log(1 / pred_days - 1);
        #pred_days = pred_days * self.division_norm;

        loss_val = loss_sum / loss_count;
        return gt_days, pred_days, loss_val;
        
    def train(self, args):
        optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.residual_loss, var_list=self.g_vars);

        print "Begin loading data";
        self.X_train, self.Y_train, self.mu, self.sigma = self.load_data(self.training_dir_list);
        self.X_val, self.Y_val, self.mu, self.sigma = self.load_data(self.val_dir_list, self.mu, self.sigma);
        print "Finish loading data";

        init_op = tf.global_variables_initializer();
        self.sess.run(init_op);

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        counter = 0;
        loss_sum = 0;
        loss_count = 0;
        print "No. traning data: ", (len(self.X_train) // self.batch_size), self.X_train.shape[0];
        for epoch in xrange(args.epoch):
            batch_idxs = len(self.X_train) // self.batch_size;

            for idx in xrange(0, batch_idxs):
                batch_images = self.X_train[idx*self.batch_size:(idx+1)*self.batch_size];
                batch_groundtruths = self.Y_train[idx*self.batch_size:(idx+1)*self.batch_size];

                # Not update parameter for the first epoch
                if (counter > self.X_train.shape[0]):
                    self.sess.run([optim], feed_dict={self.input_image: batch_images, self.raw_groundtruth: batch_groundtruths});

                loss_value = self.residual_loss.eval({self.input_image: batch_images, self.raw_groundtruth: batch_groundtruths});
                loss_sum += loss_value;
                loss_count += 1;
                
                if (counter > self.X_train.shape[0]) and (np.mod(counter, self.X_train.shape[0]) == 1):
                    print("Epoch %d - Loss %4.8f " % (epoch-1, (loss_sum / loss_count)));
                    loss_sum = 0;
                    loss_count = 0;
                    sys.stdout.flush();

                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter);

                if ((epoch >= 0) and (np.mod(counter, self.X_train.shape[0]) == 1)):
                #if (counter >= 1000000):
                    gt_days, pred_days, loss_val = self.validate();
                    print("Validation %4.8f" % (loss_val));
                    # Print results to file
                    with open("val/val_%d.txt"%(epoch), "w") as text_file:
                        for val_idx in range(gt_days.shape[0]):
                            #print "val ", gt_days[val_idx], pred_days[val_idx];
                            text_file.write("%4.4f %4.4f\n" % (gt_days[val_idx], pred_days[val_idx]));


                counter += 1;


        self.save(args.checkpoint_dir, counter);


    def save(self, checkpoint_dir, step):
        model_name = "cae.model"
        model_dir = "%s" % (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def generator(self, image):
        with tf.variable_scope("generator") as scope:
            self.gf_dim = 64;
            self.output_size = 1;
      
            """
            s_d = self.image_size[0];
            s_h = self.image_size[1];
            s_w = self.image_size[2];
            print "sd, sh, sw", s_d, s_h, s_w;
            s_d_2, s_d_4, s_d_8, s_d_16, s_d_32, s_d_64, s_d_128 = int(s_d/2), int(s_d/4), int(s_d/8), int(s_d/16), int(s_d/32), int(s_d/64), int(s_d/128);
            s_h_2, s_h_4, s_h_8, s_h_16, s_h_32, s_h_64, s_h_128 = int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16), int(s_h/32), int(s_h/64), int(s_h/128);
            s_w_2, s_w_4, s_w_8, s_w_16, s_w_32, s_w_64, s_w_128 = int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16), int(s_w/32), int(s_w/64), int(s_w/128);
            """
            # Encoding part
            print "Input's dimension: ", image.get_shape();
            e1 = tf.layers.max_pooling2d(lrelu(conv2d(image, self.gf_dim, name='g_e1_conv')), pool_size=(2,2), strides=(2,2));
            print "e1 dim ", e1.get_shape();
            e2 = self.g_bn_e2(tf.layers.max_pooling2d(conv2d(e1, self.gf_dim*2, name='g_e2_conv'), pool_size=(2,2), strides=(2,2)));
            print "e2 dim ", e2.get_shape();
            e3 = self.g_bn_e3(tf.layers.max_pooling2d(conv2d(e2, self.gf_dim*4, name='g_e3_conv'), pool_size=(2,2), strides=(2,2)));
            print "e3 dim ", e3.get_shape();
            e4 = self.g_bn_e4(tf.layers.max_pooling2d(conv2d(e3, self.gf_dim*8, name='g_e4_conv'), pool_size=(2,2), strides=(2,2)));
            print "e4 dim ", e4.get_shape();
            #e5 = conv3d(lrelu(e4), self.gf_dim*8, name='g_e5_conv');
            #e6 = conv3d(lrelu(e5), self.gf_dim*8, name='g_e6_conv');
            #e7 = conv3d(lrelu(e6), self.gf_dim*8, name='g_e7_conv');

            secondlast = self.g_bn_e5(tf.layers.max_pooling2d(e4, pool_size=(2,2), strides=(2,2)));

            self.out_layer = tf.sigmoid(fc2d(secondlast, 1, name='g_out'));
            print "out layer shape", self.out_layer.get_shape();

            return self.out_layer;


    def load_data(self, dir_list, mu=None, sigma=None):
        X_loaded = np.zeros(shape=(0, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        X_sliced = np.zeros(shape=(0, self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        Y_loaded = np.zeros(shape=(0,), dtype = np.float32);
        for data_folder in dir_list:
            X, Y, X_c = self.load_data_folder(data_folder);
            X_loaded = np.concatenate((X_loaded, X));
            X_sliced = np.concatenate((X_sliced, X_c));
            Y_loaded = np.concatenate((Y_loaded, Y));

        if (not os.path.exists('area_label.txt')):
            X_area = np.sum(X_loaded, axis=(1,2,3,4));
            self.save_area_label(X_area, Y_loaded);

        #X_flatten = np.reshape(X_train, (-1));
        #X_fg_idx = np.where(X_flatten > 0.5);
        #X_fg = X_flatten(X_fg_idx);

        #mu = np.mean(X_fg);
        #sigma = np.std(X_fg);
        X_sliced = X_sliced / 409;
        if (mu == None):
            mu = np.mean(X_sliced.flatten());
            sigma = np.std(X_sliced.flatten());
            print "mu, sigma: ", mu, sigma;
            print "maxX, minX: ", np.amax(X_sliced.flatten()), np.amin(X_sliced.flatten());
        
        X_sliced = (X_sliced - mu) / sigma;

        #print "After normalizing, max, min: ", np.amax(X_train.flatten()), np.amin(X_train.flatten());

        return X_sliced, Y_loaded, mu, sigma;

    def load_data_folder(self, folder):
        X_train = np.zeros(shape=(800, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        X_slice = np.zeros(shape=(800, self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        Y_train = np.zeros(shape=(800,), dtype=np.float32);

        # Load info.txt for name matching
        with open(folder + '/info.txt') as f:
                content = f.readlines();
        content = [x.strip() for x in content];
        dict_name = {};
        for line in content:
            org, des_idx = line.split(' ');
            des = 'img_' + str(des_idx);
            dict_name[des] = org;

        # Load clinical data file
        with open(folder + '/../survival_label.txt') as f:
                content = f.readlines();
        content = [x.strip() for x in content];
        headers = content[0].split(',');
        survival_index = headers.index('\"Survival.time\"');
        dict_survival = {};
        for line in content[1::]:
            parts = line.split(',');
            org_trim = parts[0].replace('\"', '');
            survival_trim = int(parts[survival_index]);
            dict_survival[org_trim] = survival_trim;


        idx = 0;
        idx_debug = 0;
        for file_name in glob.glob(folder + '/*.mat'):
            loaded = sio.loadmat(file_name);
            img = loaded['norm_tumor'];
            img = np.transpose(img, (2, 0, 1));
            X_train[idx, :, :, :, 0] = img;
            _, filename_tail = os.path.split(file_name)
            Y_train[idx] = dict_survival[dict_name[os.path.splitext(filename_tail)[0]]];

            # Select the biggest slice
            print "img dim ", img.shape;
            X_sum = np.sum(img, axis=(1,2));
            print "X_sum ", X_sum.shape;
            max_slice_idx = np.argmax(X_sum);
            X_slice[idx, :, :, 0] = img[max_slice_idx, :, :];

            if (idx_debug == 0):
                idx_debug = 1;
                img_dir = os.path.splitext(filename_tail)[0];
                if (not os.path.exists(img_dir)):
                    os.makedirs(img_dir);
                    sample_path = img_dir + '/selected.jpg';
                    scipy.misc.imsave(sample_path, X_slice[idx, :, :, 0]/4095);
                    """
                    for i_slice in range(img.shape[0]):
                        slices = img[i_slice,:,:];
                        slices = slices / 4095;
                        sample_path = img_dir + '/' + str(i_slice) + '.jpg';
                        print "sample path ", sample_path;
                        scipy.misc.imsave(sample_path, slices);
                    """
            idx += 1;

            #if (np.amin(img.flatten()) < 0):
                #print file_name;

        X_train = X_train[:idx];
        X_slice = X_slice[:idx];
        Y_train = Y_train[:idx];

        return X_train, Y_train, X_slice;

    def save_area_label(self, x, y):
        with open('area_label.txt', "w") as text_file:
            for idx in range(x.shape[0]):
                text_file.write('%.4f %f\n'%(x[idx], y[idx]));


    def test(self):
        init_op = tf.global_variables_initializer();
        self.sess.run(init_op);

        synthesis_input = np.random.rand(1, self.image_size[0], self.image_size[1], self.image_size[2], 1);
        samples = self.sess.run(self.reconstructed, feed_dict={self.input_image: synthesis_input});
        print "samples shape ", samples.shape;
        print samples[0, 1:3, 1:3, 1:3, 0];




