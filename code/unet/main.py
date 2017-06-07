import argparse
import os
import scipy.misc
import numpy as np

from model import CAE_3D
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')


args = parser.parse_args()


def main(_):
    with tf.Session() as sess:
        model = CAE_3D(sess);

        model.test();



if __name__ == '__main__':
        tf.app.run()
