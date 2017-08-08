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

        # Param of classification
        self.survival_thresh = 400;

        #self.clinical_numeric_fields = ['age', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage'];
        self.clinical_numeric_fields = ['age', 'clinical.T.Stage', 'Clinical.N.Stage'];
        self.clinical_category_fields = ['gender'];


        self.division_norm = 700;      # This number is used to divide to norm the output to 0 --> 1 range
        self.batch_size = 1;
        self.image_size = [64, 128, 128, 1];      # [depth, height, width, channel]
        self.dataset_name = dataset_name;
        self.checkpoint_dir = checkpoint_dir;

        self.training_dir_list = ['/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_61_180_tumor', \
                                    '/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_181_422_tumor'];
        self.val_dir_list = ['/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_60_tumor'];

        self.g_bn_e1 = batch_norm(name='g_bn_e1')
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        #self.g_bn_e7 = batch_norm(name='g_bn_e7')
        #self.g_bn_e8 = batch_norm(name='g_bn_e8')


        self.build_model();

    def build_model(self):
        self.input_image = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]], name='input_image');
        self.clinical_numeric = tf.placeholder(tf.float32, [1, 1, 1, 1, len(self.clinical_numeric_fields)], name='clinical_numeric');
        self.output_certainty = tf.placeholder(tf.float32, [self.batch_size], name='output_certainty');
        self.raw_groundtruth = tf.placeholder(tf.float32, [self.batch_size, 2], name='raw_output');
        #self.norm_groundtruth = tf.sigmoid(self.raw_groundtruth / self.division_norm);

        self.reconstruct = self.generator(self.input_image, self.clinical_numeric);

        #print "loss output: GT, generated: ", self.raw_groundtruth.get_shape(), self.pred_survival.get_shape();

        #self.residual_loss = tf.reduce_mean(tf.abs(self.pred_survival - self.norm_groundtruth));
        #self.residual_loss = self.output_certainty * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.raw_groundtruth, logits=self.pred_survival, dim=-1));
        self.reconstruct_flat = tf.reshape(self.reconstruct, [self.batch_size, -1]);
        self.input_flat = tf.reshape(self.input_image, [self.batch_size, -1]);
        self.reconstruct_loss = tf.reduce_mean(tf.abs(self.reconstruct - self.input_image));
        print "input_flat shape ", self.input_image.get_shape();
        print "reconstruct_flat shape ", self.reconstruct.get_shape();
        print "reconstruct_loss shape ", self.reconstruct_loss.get_shape();

        t_vars = tf.trainable_variables();
        self.g_vars = [var for var in t_vars if 'g_' in var.name];
        print "No. of trained vars: ", len(self.g_vars);
        for g_v in self.g_vars:
            print g_v.name;

        self.saver = tf.train.Saver();

    def validate(self):
        # Additional printed values
        gt_days_raw = np.zeros(shape=(self.X_val.shape[0],), dtype=np.float32);
        name_arr = [];

        batch_idxs = len(self.X_val) // self.batch_size;
        loss_sum = 0;
        loss_count = 0;

        """
        with tf.variable_scope('generator/g_fc1', reuse=True):
            w = tf.get_variable('w', [4, 8, 8, 512, 512]);
            weight = self.sess.run(w);
            sum_weight = np.sum(weight.flatten());
            print "sum weight ", sum_weight;
        """

        for idx in xrange(0, batch_idxs):
            batch_images = self.X_val[idx*self.batch_size:(idx+1)*self.batch_size];
            batch_name = self.name_val[idx*self.batch_size:(idx+1)*self.batch_size];

            #lay1, lay2, loss_value, normgroundtruth, pred_sur = self.sess.run([self.net['fc1'], self.net['fc2'], self.residual_loss, self.norm_groundtruth, self.pred_survival], \
                                                #feed_dict={self.input_image: batch_images, self.raw_groundtruth: batch_groundtruths});
            recons, loss_value, normgroundtruth, pred_sur = self.sess.run([self.net['output'], self.residual_loss, self.raw_groundtruth, self.pred_survival], \
                    feed_dict={self.input_image: batch_images, self.raw_groundtruth: batch_groundtruths, \
                                self.output_certainty: batch_certainty, self.clinical_numeric: batch_clinical_num});
            name_arr.append(batch_name[0]);
            loss_sum += loss_value;
            loss_count += 1;

        #pred_days = np.log(1 / pred_days - 1);
        #pred_days = pred_days * self.division_norm;

        loss_val = loss_sum / loss_count;
        return gt_days, pred_days, loss_val, gt_days_raw, name_arr;
        
    def train(self, args):
        optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.reconstruct_loss, var_list=self.g_vars);

        print "Begin loading data";
        self.X_train, self.Y_train, self.Y_train_raw, self.certainty_train, self.clinical_num_train, self.name_train, self.mu, self.sigma = self.load_data(self.training_dir_list);
        self.X_val, self.Y_val, self.Y_val_raw, self.certainty_val, self.clinical_num_val, self.name_val, self.mu, self.sigma = self.load_data(self.val_dir_list, self.mu, self.sigma);
        #self.X_val, self.Y_val, self.certainty_val = self.filter_certain_sample(self.X_val, self.Y_val, self.certainty_val);
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
                batch_certainty = self.certainty_train[idx*self.batch_size:(idx+1)*self.batch_size];
                batch_clinical_num = self.clinical_num_train[idx*self.batch_size:(idx+1)*self.batch_size];

                # Not update parameter for the first epoch
                if (counter > self.X_train.shape[0]):
                    self.sess.run([optim], feed_dict={self.input_image: batch_images, self.raw_groundtruth: batch_groundtruths, \
                                                        self.output_certainty: batch_certainty, self.clinical_numeric: batch_clinical_num});

                loss_value = self.reconstruct_loss.eval({self.input_image: batch_images, self.raw_groundtruth: batch_groundtruths, \
                                                        self.output_certainty: batch_certainty, self.clinical_numeric: batch_clinical_num});
                loss_sum += loss_value;
                loss_count += 1;
                
                if (counter > self.X_train.shape[0]) and (np.mod(counter, self.X_train.shape[0]) == 1):
                    print("Epoch %d - Loss %4.8f " % (epoch-1, (loss_sum / loss_count)));
                    loss_sum = 0;
                    loss_count = 0;
                    sys.stdout.flush();

                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter);

                """
                if ((epoch >= 0) and (np.mod(counter, self.X_train.shape[0]) == 1)):
                #if (counter >= 1000000):
                    gt_days, pred_days, loss_val, gt_days_raw, name_arr = self.validate();
                    gt_days   = self.convert_vector_to_binary(gt_days);
                    pred_days = self.convert_vector_to_binary(pred_days);
                    intersect = 1*(gt_days == pred_days);
                    acc_val = np.sum(intersect) / pred_days.shape[0];
                    print("Validation Loss %4.8f" % (loss_val));
                    print("Validation Acc %4.8f" % (acc_val));
                    # Print results to file
                    with open("val/val_%d.txt"%(epoch), "w") as text_file:
                        for val_idx in range(gt_days.shape[0]):
                            #print "val ", gt_days[val_idx], pred_days[val_idx];
                            text_file.write("%s %4.4f %4.4f %4.4f\n" % (name_arr[val_idx], gt_days[val_idx], pred_days[val_idx], gt_days_raw[val_idx]));
                """

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


    def generator(self, image, clinical_num):
        with tf.variable_scope("generator") as scope:
            self.gf_dim = 64;
            self.output_size = 1;
            
            s_d = self.image_size[0];
            s_h = self.image_size[1];
            s_w = self.image_size[2];
            s_d_2, s_d_4, s_d_8, s_d_16, s_d_32, s_d_64, s_d_128 = int(s_d/2), int(s_d/4), int(s_d/8), int(s_d/16), int(s_d/32), int(s_d/64), int(s_d/128);
            s_h_2, s_h_4, s_h_8, s_h_16, s_h_32, s_h_64, s_h_128 = int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16), int(s_h/32), int(s_h/64), int(s_h/128);
            s_w_2, s_w_4, s_w_8, s_w_16, s_w_32, s_w_64, s_w_128 = int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16), int(s_w/32), int(s_w/64), int(s_w/128);

            s_deconv_d = 8;
            s_deconv_h = 16;
            s_deconv_w = 16;

            net = {};
            net['conv1'] = lrelu(conv3d(image, self.gf_dim/2, name='g_conv1'));
            net['pool1'] = tf.layers.max_pooling3d(net['conv1'], pool_size=(2,2,2), strides=(2,2,2));

            net['conv2'] = lrelu(conv3d(net['pool1'], self.gf_dim, name='g_conv2'));
            net['pool2'] = tf.layers.max_pooling3d(net['conv2'], pool_size=(2,2,2), strides=(2,2,2));

            net['conv3'] = lrelu(conv3d(net['pool2'], self.gf_dim*2, name='g_conv3'));
            net['pool3'] = tf.layers.max_pooling3d(net['conv3'], pool_size=(2,2,2), strides=(2,2,2));
            """
            net['conv4'] = lrelu(conv3d(net['pool3'], self.gf_dim*2, name='g_conv4'));
            net['pool4'] = tf.layers.max_pooling3d(net['conv4'], pool_size=(2,2,2), strides=(2,2,2));

            net['deconv1'], _, _ = deconv3d(tf.nn.relu(net['pool4']),   [self.batch_size, s_deconv_d*1, s_deconv_h*1, s_deconv_w*1, self.gf_dim*2], d_d=2, d_h=2, d_w=2, stddev=0.02, name="g_deconv1", with_w=True);
            """
            net['deconv2'], _, _ = deconv3d(tf.nn.relu(net['pool3']), [self.batch_size, s_deconv_d*2, s_deconv_h*2, s_deconv_w*2, int(self.gf_dim)], d_d=2, d_h=2, d_w=2, stddev=0.02, name="g_deconv2", with_w=True);
            net['deconv3'], _, _ = deconv3d(tf.nn.relu(net['deconv2']), [self.batch_size, s_deconv_d*4, s_deconv_h*4, s_deconv_w*4, int(self.gf_dim/2)], d_d=2, d_h=2, d_w=2, stddev=0.02, name="g_deconv3", with_w=True);
            net['deconv4'], _, _ = deconv3d(tf.nn.relu(net['deconv3']), [self.batch_size, s_deconv_d*8, s_deconv_h*8, s_deconv_w*8, 1], d_d=2, d_h=2, d_w=2, stddev=0.02, name="g_deconv4", with_w=True);
            
            """
            net['fc1'] = conv3d(net['pool3'], self.gf_dim*2, k_d=8, k_h=16, k_w=16, pad_='VALID', name='g_fc1');    # ??? originally self.gf_dim*8
            print "fc1 dim ", net['fc1'].get_shape();

            net['clinical_num_fc1'] = conv3d(clinical_num, 3*len(self.clinical_numeric_fields), k_d=1, k_h=1, k_w=1, pad_='VALID', name='g_cli_num_fc1');
            net['concat1'] = tf.concat([net['fc1'], net['clinical_num_fc1']], axis=4, name='g_concat1');

            net['fc2'] = conv3d(net['concat1'], 2, k_d=1, k_h=1, k_w=1, pad_='VALID', name='g_fc2');
            print "fc2 dim ", net['fc2'].get_shape();

            net['output'] = tf.reshape(net['fc2'], shape=(self.batch_size, 2));
            print "Output dim ", net['output'].get_shape();
            """

            net['output'] = net['deconv4'];

            self.net = net;


            """
            # Encoding part
            print "Input's dimension: ", image.get_shape();
            e1 = lrelu(conv3d(image, self.gf_dim, name='g_e1_conv'));
            e2 = self.g_bn_e2(tf.layers.max_pooling3d(lrelu(conv3d(e1, self.gf_dim*2, name='g_e2_conv')), pool_size=(2,2,2), strides=(2,2,2)));
            e3 = self.g_bn_e3(lrelu(conv3d(e2, self.gf_dim*4, name='g_e3_conv')));
            e4 = self.g_bn_e4(tf.layers.max_pooling3d(lrelu(conv3d(e3, self.gf_dim*8, name='g_e4_conv')), pool_size=(2,2,2), strides=(2,2,2)));
            e5 = self.g_bn_e5(lrelu(conv3d(e4, self.gf_dim*8, name='g_e5_conv')));
            e6 = self.g_bn_e6(tf.layers.max_pooling3d(lrelu(conv3d(e5, self.gf_dim*8, name='g_e6_conv')), pool_size=(2,2,2), strides=(2,2,2)));

            #secondlast = self.g_bn_e5(tf.layers.max_pooling3d(e4, pool_size=(2,2,2), strides=(2,2,2)));

            print "e6 dim ", e6.get_shape();
            e7 = self.g_bn_e7(lrelu(conv3d(e6, self.gf_dim*8, k_d=8, k_h=16, k_w=16, pad_='VALID', name='g_e7_conv')));
            print "e7 dim ", e7.get_shape();
            #self.out_layer = tf.sigmoid(fc(secondlast, 1, name='g_out'));
            self.out_layer = tf.sigmoid(fc3d(e7, 1, name='g_out'));
            print "out layer shape", self.out_layer.get_shape();
            """

            return net['output'];

    def synthesize_data():
        X_syn = np.zeros(shape=(1000, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        Y_syn = np.zeros(shape=(1000,), dtype = np.float32);

        #for idx in range(X_syn.shape[0]):


    def load_data(self, dir_list, mu=None, sigma=None):
        X_loaded = np.zeros(shape=(0, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        Y_raw_loaded = np.zeros(shape=(0,), dtype = np.float32);
        D_loaded = np.zeros(shape=(0,), dtype = np.float32);
        A_num_loaded = np.zeros(shape=(0, len(self.clinical_numeric_fields)), dtype=np.float32);       # Additional clinical information as input
        Name_loaded = [];
        
        for data_folder in dir_list:
            X, Y, D, A_num, N = self.load_data_folder(data_folder);
            X_loaded = np.concatenate((X_loaded, X));
            Y_raw_loaded = np.concatenate((Y_raw_loaded, Y));
            D_loaded = np.concatenate((D_loaded, D));
            A_num_loaded = np.concatenate((A_num_loaded, A_num));
            Name_loaded = Name_loaded + N;

        # Generate Y_loaded and label_certainty from Y_raw_loaded
        label_certainty = np.ones(shape=D_loaded.shape, dtype=np.float32);
        print "Y_raw_loaded, D_loaded shape ", Y_raw_loaded.shape, D_loaded.shape;
        for i in range(D_loaded.shape[0]):
            if ((Y_raw_loaded[i] < self.survival_thresh) and (D_loaded[i] > 0)):
                #label_certainty[i] = Y_raw_loaded[i] / self.survival_thresh;
                label_certainty[i] = 1.0;       # ???

        # Reshape A_num_loaded to fit the network's input
        A_num_loaded = np.reshape(A_num_loaded, newshape=(A_num_loaded.shape[0],1,1,1,-1));

        #X_flatten = np.reshape(X_train, (-1));
        #X_fg_idx = np.where(X_flatten > 0.5);
        #X_fg = X_flatten(X_fg_idx);

        #mu = np.mean(X_fg);
        #sigma = np.std(X_fg);
        X_loaded = X_loaded / 409;
        if (mu == None):
            mu = np.mean(X_loaded.flatten());
            sigma = np.std(X_loaded.flatten());
            print "mu, sigma: ", mu, sigma;
            print "maxX, minX: ", np.amax(X_loaded.flatten()), np.amin(X_loaded.flatten());
        
        X_loaded = (X_loaded - mu) / sigma;

        #print "After normalizing, max, min: ", np.amax(X_train.flatten()), np.amin(X_train.flatten());
        Y_loaded = self.convert_survival_to_vector(Y_raw_loaded, self.survival_thresh);

        return X_loaded, Y_loaded, Y_raw_loaded, label_certainty, A_num_loaded, Name_loaded, mu, sigma;

    def filter_certain_sample(self, X, Y, Cer):
        X_new =  np.zeros(shape=(0, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        Y_new = np.zeros(shape=(0, 2), dtype = np.float32);
        C_new = np.zeros(shape=(0,), dtype = np.float32);

        for idx in range(Cer.shape[0]):
            if (Cer[idx] >= 0.98):
                X_new = np.insert(X_new, X_new.shape[0], X[idx], axis=0);
                Y_new = np.insert(Y_new, Y_new.shape[0], Y[idx], axis=0);
                C_new = np.insert(C_new, C_new.shape[0], Cer[idx], axis=0);

        return X_new, Y_new, C_new;

    def convert_vector_to_binary(self, vector):
        bin_array = np.zeros(shape=(vector.shape[0],), dtype = np.float32);
        print "output vector shape: ", vector.shape;
        for idx in range(vector.shape[0]):
            if (vector[idx][0] > vector[idx][1]):
                bin_array[idx] = 0;
            else:
                bin_array[idx] = 1;

        return bin_array;

    def convert_survival_to_vector(self, survival_arr, threshold):
        survival_vec = np.zeros(shape=(survival_arr.shape[0],2), dtype = np.float32);
        for idx in range(survival_arr.shape[0]):
            if (survival_arr[idx] > threshold):
                survival_vec[idx][1] = 1;
            else:
                survival_vec[idx][0] = 1;

        return survival_vec;

    def load_data_folder(self, folder):
        Name_train = [];
        X_train = np.zeros(shape=(800, self.image_size[0], self.image_size[1], self.image_size[2], self.image_size[3]), dtype = np.float32);
        Y_train = np.zeros(shape=(800,), dtype=np.float32);     # Label
        D_train = np.zeros(shape=(800,), dtype=np.float32);     # Deadstatus
        A_num_train = np.zeros(shape=(800, len(self.clinical_numeric_fields)), dtype=np.float32);       # Additional clinical information as input

        # Load info.txt for name matching
        with open(folder + '/info.txt') as f:
                content = f.readlines();
        content = [x.strip() for x in content];
        dict_name = {};
        for line in content:
            org, des_idx = line.split(' ');
            des = 'img_' + str(des_idx);
            dict_name[des] = org;


        # Load deadstatus and additional clinical information
        dict_survival = {};
        dict_deadstat = {};
        dict_clinical_num = {};
        with open(folder + '/../survival_label.txt') as f:
                content = f.readlines();
        content = [x.strip() for x in content];
        headers = content[0].split(',');
        survival_index = headers.index('\"Survival.time\"');
        deadstat_index = headers.index('\"deadstatus.event\"');
        clinical_num_index = [0] * len(self.clinical_numeric_fields);
        for i_cli in range(len(self.clinical_numeric_fields)):
            clinical_num_index[i_cli] = headers.index('\"' + self.clinical_numeric_fields[i_cli] + '\"');
            dict_clinical_num[self.clinical_numeric_fields[i_cli]] = {};

        for line in content[1::]:
            parts = line.split(',');
            org_trim = parts[0].replace('\"', '');
            survival_trim = float(parts[survival_index]);
            deadstat_trim = int(parts[deadstat_index]);
            dict_survival[org_trim] = survival_trim;
            dict_deadstat[org_trim] = deadstat_trim;
            # Clinical data
            for i_cli in range(len(self.clinical_numeric_fields)):
                """
                if (org_trim == 'LUNG1-056'):
                    print parts[clinical_num_index[i_cli]];
                    print "Diff ", (parts[clinical_num_index[i_cli]] != '\"NA\"');
                    print "Equal ", (parts[clinical_num_index[i_cli]] == '\"NA\"');
                    if (parts[clinical_num_index[i_cli]] != '\"NA\"'):
                        print "aaaaaa";
                    else:
                        print "bbbbbb";
                """
                if (parts[clinical_num_index[i_cli]] != '\"NA\"'):
                    dict_clinical_num[self.clinical_numeric_fields[i_cli]][org_trim] = float(parts[clinical_num_index[i_cli]]);
                else:
                    if (self.clinical_numeric_fields[i_cli] == "age"):
                        dict_clinical_num[self.clinical_numeric_fields[i_cli]][org_trim] = 68.0;    # This is the average value
                    else:
                        dict_clinical_num[self.clinical_numeric_fields[i_cli]][org_trim] = None;




        idx = 0;
        idx_debug = 0;
        for file_name in glob.glob(folder + '/*.mat'):
            _, filename_tail = os.path.split(file_name)

            if (idx == 23):
                clinical_field_name = self.clinical_numeric_fields[0];
                print "23";
                print dict_name[os.path.splitext(filename_tail)[0]];
                print dict_clinical_num[clinical_field_name][dict_name[os.path.splitext(filename_tail)[0]]];
                print (dict_clinical_num[clinical_field_name][dict_name[os.path.splitext(filename_tail)[0]]] == None);

            # Assign clinical numeric values
            is_None = False;
            for i_cli in range(len(self.clinical_numeric_fields)):
                clinical_field_name = self.clinical_numeric_fields[i_cli];
                a_value = dict_clinical_num[clinical_field_name][dict_name[os.path.splitext(filename_tail)[0]]];
                if (a_value == None):
                    is_None = True;
                else:
                    A_num_train[idx][i_cli] = a_value;
                
                #if (A_num_train[idx][i_cli] == None):
                    #is_None = True;

                """
                if ((idx == 23) and (i_cli == 0)):
                    print dict_clinical_num[clinical_field_name][dict_name[os.path.splitext(filename_tail)[0]]];
                    print A_num_train[idx][i_cli];
                    print (A_num_train[idx][i_cli] == None);
                """
            """
            if (idx == 23):
                print is_None;
                print (dict_clinical_num[self.clinical_numeric_fields[0]][dict_name[os.path.splitext(filename_tail)[0]]] == None);
                print A_num_train[idx];
            """

            # If any field has 'NA' value, skip this sample
            if (is_None == True):
                continue;



            loaded = sio.loadmat(file_name);
            img = loaded['norm_tumor'];
            img = np.transpose(img, (2, 0, 1));
            X_train[idx, :, :, :, 0] = img;
            Y_train[idx] = dict_survival[dict_name[os.path.splitext(filename_tail)[0]]];
            D_train[idx] = dict_deadstat[dict_name[os.path.splitext(filename_tail)[0]]];
            Name_train.append(dict_name[os.path.splitext(filename_tail)[0]]);


            """
            if (idx_debug == 0):
                idx_debug = 1;
                img_dir = os.path.splitext(filename_tail)[0];
                os.makedirs(img_dir);
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
        Y_train = Y_train[:idx];
        D_train = D_train[:idx];
        A_num_train = A_num_train[:idx];

        return X_train, Y_train, D_train, A_num_train, Name_train;


    def test(self):
        init_op = tf.global_variables_initializer();
        self.sess.run(init_op);

        synthesis_input = np.random.rand(1, self.image_size[0], self.image_size[1], self.image_size[2], 1);
        samples = self.sess.run(self.reconstructed, feed_dict={self.input_image: synthesis_input});
        print "samples shape ", samples.shape;
        print samples[0, 1:3, 1:3, 1:3, 0];




