import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from CommonUtil import logger
from Trainer import Trainer


def main(datadir, dataset_start_idx, dataset_end_idx, data_use_num,
         sp=256, first_layer_ch=24,
         batch_size=8, epoch_num=40,
         discriminator_out_res=32,
         mode='retrain', pre_model_dir=None):
    root_dir = os.getcwd()
    graph_dir = os.path.join(root_dir, 'graph%dp' % sp)
    results_dir = os.path.join(root_dir, 'results%dp' % sp)
    code_bk_dir = os.path.join(results_dir, 'code_bk')
    if mode == 'retrain' or mode == 'finetune':
        import shutil
        if os.path.exists(graph_dir): shutil.rmtree(graph_dir)
        if os.path.exists(results_dir): shutil.rmtree(results_dir)
        os.mkdir(results_dir)
        os.mkdir(code_bk_dir)

        # backup source code
        file_list = os.listdir(root_dir)
        for item in file_list:
            full_name = os.path.join(root_dir, item)
            if os.path.isfile(full_name) and item.endswith('.py'):
                shutil.copy(full_name, os.path.join(code_bk_dir, item))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    logger.set_log_file(os.path.join(results_dir, 'log.txt'))
    logger.write('Constructing network graph...')

    trainer = Trainer(sess, sp, discriminator_out_res)
    trainer.train(datadir, dataset_start_idx, dataset_end_idx, data_use_num, 24, first_layer_ch,
                  results_dir=results_dir, graph_dir=graph_dir,
                  batch_size=batch_size, epoch_num=epoch_num,
                  mode=mode, pre_model_dir=pre_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='path to the training dataset')
    parser.add_argument('--dataset_start_idx', type=int, required=True, help='beginning index of the data')
    parser.add_argument('--dataset_end_idx', type=int, required=True, help='ending index of the data')
    parser.add_argument('--data_use_num', type=int, required=True, help='number of data pairs used in training')
    parser.add_argument('--sp', type=int, default=256, help='generator input/output resolution')
    parser.add_argument('--first_layer_ch', type=int, default=24, help='The channel number of the first encoder layer')
    parser.add_argument('--batch_size', type=int, default=8, help='size of a training batch')
    parser.add_argument('--epoch_num', type=int, default=40, help='number of training epochs')
    parser.add_argument('--disc_out_res', type=int, default=32, help='number of patch per row/column')
    parser.add_argument('--mode', type=str, default='retrain', choices=['retrain', 'resume', 'finetune'], help='mode')
    parser.add_argument('--pre_model_dir', type=str, default=None, help='path of the model to be loaded')

    args = parser.parse_args()
    main(args.datadir, args.dataset_start_idx, args.dataset_end_idx, args.data_use_num,
         args.sp, args.first_layer_ch,
         args.batch_size, args.epoch_num,
         args.disc_out_res,
         args.mode, args.pre_model_dir)


