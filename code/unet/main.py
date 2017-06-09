import argparse
import os
import scipy.misc
import numpy as np

from model import CAE_3D
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='nsclc', help='name of the dataset')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')

args = parser.parse_args()


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        model = CAE_3D(sess, dataset_name=args.dataset_name, checkpoint_dir=args.checkpoint_dir);

        model.train(args);
        #model.test();



if __name__ == '__main__':
        tf.app.run()
