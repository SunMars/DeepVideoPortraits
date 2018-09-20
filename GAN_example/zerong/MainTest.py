import os
import sys
import json
import argparse
import time
import shutil
import numpy as np
import tensorflow as tf

from CommonUtil import logger
from Trainer import Trainer


def main(datadir, dataset_start_idx, dataset_end_idx,
         sp=256, first_layer_ch=24,
         discriminator_out_res=32,
         model_dir='./results512p'):
    root_dir = os.getcwd()
    test_dir = os.path.join(root_dir, 'test%dp' % sp)
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    os.mkdir(os.path.join(test_dir, 'gt'))
    os.mkdir(os.path.join(test_dir, 'input'))
    os.mkdir(os.path.join(test_dir, 'output'))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    trainer = Trainer(sess, sp, discriminator_out_res)
    trainer.test(datadir, dataset_start_idx, dataset_end_idx, 24, first_layer_ch,
                 results_dir=test_dir, model_dir=model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='path to the training dataset')
    parser.add_argument('--dataset_start_idx', type=int, required=True, help='beginning index of the data')
    parser.add_argument('--dataset_end_idx', type=int, required=True, help='ending index of the data')
    parser.add_argument('--sp', type=int, default=256, help='generator input/output resolution')
    parser.add_argument('--first_layer_ch', type=int, default=24, help='The channel number of the first encoder layer')
    parser.add_argument('--disc_out_res', type=int, default=32, help='number of patch per row/column')
    parser.add_argument('--model_dir', type=str, default=None, help='path of the model to be loaded')

    args = parser.parse_args()
    main(args.datadir, args.dataset_start_idx, args.dataset_end_idx,
         args.sp, args.first_layer_ch,
         args.disc_out_res,
         args.model_dir)


