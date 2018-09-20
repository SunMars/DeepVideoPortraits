import os

import numpy as np
import tensorflow as tf
import scipy

from CommonUtil import logger
from DataLoader import DataLoader
from pretrainedVgg import build_pretrained_vgg
from NetUtil import Generator, Discriminator, compute_error, get_num_params


class Trainer(object):
    def __init__(self, sess, sp=256, discriminator_out_res=32):
        self.sess = sess
        self.sp = sp
        self.discriminator_out_res = discriminator_out_res

        self.G, self.Gout, self.Gout_scaled = None, None, None
        self.D_fake, self.D_fake_out, self.D_real, self.D_real_out = None, None, None, None

        self.l1_diff_map, self.attention_map = None, None
        self.G_content_loss, self.G_loss, self.D_loss = None, None, None

        self.G_content_opt, self.D_opt, self.G_total_opt = None, None, None
        self.merged_loss_scalar, self.merged_testing, self.writer = None, None, None

        self.saver = None

    def train(self,
              dataset_dir, dataset_start_idx, dataset_end_idx, data_use_num,    # dataset
              in_channel,                       # network input channel (rendered image + body-part masks)
              first_layer_channel,              # channel number of the first encoder layer
              results_dir='./results',          # results dir
              graph_dir='./graph',              # dir for tf.summary
              batch_size=8, epoch_num=40,       # training argument
              mode='retrain',                   # training mode ('retrain', 'resume' or 'finetune'
              pre_model_dir=None):              # dir of the model to load (only for 'resume' or 'finetune')
        with tf.name_scope('input'):
            with tf.variable_scope('input'):
                nin = tf.placeholder(tf.float32, [batch_size, self.sp, self.sp, in_channel])
                real_image = tf.placeholder(tf.float32, [batch_size, self.sp, self.sp, 3])
                lr = tf.placeholder(tf.float32)

        # setup
        self._setup_model(first_layer_channel, nin, real_image)
        self._setup_losses(real_image)
        self._setup_optimizer(lr)
        self._setup_summary(graph_dir, batch_size)
        self.sess.run(tf.global_variables_initializer())
        logger.write('The number of all trainable variable: %d' % get_num_params())

        # load pre-trained model to fine-tune or resume training
        if mode == 'resume':
            ckpt_prev = tf.train.get_checkpoint_state(pre_model_dir)
            if ckpt_prev:
                saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()])
                saver.restore(self.sess, ckpt_prev.model_checkpoint_path)
                logger.write('Loaded model %s' % pre_model_dir)
            else:
                logger.write('Unable to load the pretrained model. ')
        if mode == 'finetune':
            ckpt_prev = tf.train.get_checkpoint_state(pre_model_dir)
            if ckpt_prev:
                saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()
                                                 if not var.name.startswith('G/encoder_in')
                                                 and not var.name.startswith('G/decoder_out')
                                                 and not var.name.startswith('G/encoder_%dx%d' % (self.sp, self.sp))
                                                 and not var.name.startswith('G/decoder_%dx%d' % (self.sp, self.sp))
                                                 and not var.name.startswith('D/encoder_in')
                                                 and not var.name.startswith('D/encoder_%dx%d' % (self.sp, self.sp))])
                saver.restore(self.sess, ckpt_prev.model_checkpoint_path)
                logger.write('Loaded model %s' % pre_model_dir)
            else:
                logger.write('Unable to load the pretrained model. ')

        self.saver = tf.train.Saver(max_to_keep=1000)

        # initialize data loader
        data_indices = np.random.permutation(range(dataset_start_idx, dataset_end_idx))
        data_indices_training = data_indices[0:data_use_num]
        data_indices_testing = data_indices[data_use_num:data_use_num+batch_size]

        logger.write('Initializing data loader. ')
        data_loader = DataLoader(dataset_dir, self.sp, self.sp, np.max(data_indices_training))
        data_loader.preload_data(data_indices_training, 85)
        batch_num = data_use_num // batch_size

        # prepare testing batch
        test_input = [None] * batch_size
        test_target = [None] * batch_size
        for i, ind in enumerate(data_indices_testing):
            test_input[i], test_target[i] = data_loader.get_data_pair(ind)
            scipy.misc.toimage(test_target[i][0, :, :, :], cmin=0, cmax=255).save(
                '%s/testing_target_%d.png' % (results_dir, i))
        test_input_batch = np.concatenate(tuple(test_input), axis=0)
        test_target_batch = np.concatenate(tuple(test_target), axis=0)
        train_input_batch = np.zeros((batch_size, self.sp, self.sp, in_channel))
        train_target_batch = np.zeros((batch_size, self.sp, self.sp, 3))

        for epoch in range(0, epoch_num):
            logger.write('Running epoch No.%d' % epoch)
            random_indices = np.random.permutation(data_indices_training)
            random_indices = np.reshape(random_indices[0:batch_num*batch_size], (-1, batch_size))

            for bid, batch_indice in zip(range(batch_num), random_indices):
                for i, ind in enumerate(batch_indice):
                    data_pair_0, data_pair_1 = data_loader.get_data_pair(ind)
                    train_input_batch[i, :, :, :] = data_pair_0[0, :, :, :]
                    train_target_batch[i, :, :, :] = data_pair_1[0, :, :, :]

                lrate = 1e-3 if epoch < int(0.2*epoch_num) else 2e-4

                self.sess.run(self.D_opt, feed_dict={nin: train_input_batch,
                                                     real_image: train_target_batch,
                                                     lr: lrate})
                self.sess.run(self.G_total_opt, feed_dict={nin: train_input_batch,
                                                           real_image: train_target_batch,
                                                           lr: lrate})

                g_content_loss_curr, g_loss_curr, d_loss_curr, graph_results = \
                    self.sess.run([self.G_content_loss, self.G_loss, self.D_loss, self.merged_loss_scalar],
                                  feed_dict={nin: train_input_batch, real_image: train_target_batch})
                logger.write(
                    'Epoch No.%d, Batch No.%d: G_loss:%.4f, G_content_loss:%.4f, G_disc_loss:%.4f, D_loss:%.4f'
                    % (epoch, bid, g_loss_curr, g_content_loss_curr, g_loss_curr-g_content_loss_curr, d_loss_curr))
                self.writer.add_summary(graph_results, epoch * batch_num + bid)

                if bid % 10 == 0:
                    testing_results = self.sess.run(self.merged_testing,
                                                    feed_dict={nin: test_input_batch, real_image: test_target_batch})
                    self.writer.add_summary(testing_results)

                if bid == batch_num-1:
                    logger.write('End of epoch. ')
                    # save testing output
                    os.makedirs(os.path.join(results_dir, '%04d' % epoch))
                    self.saver.save(self.sess, os.path.join(results_dir, 'model.ckpt'))
                    output = self.sess.run(self.Gout_scaled, feed_dict={nin: test_input_batch, real_image: test_target_batch})
                    for i in range(batch_size):
                        scipy.misc.toimage(output[i, :, :, :], cmin=0, cmax=255).save(
                            '%s/%04d/%06d_output.png' % (results_dir, epoch, i))

                    # save model
                    if epoch % 10 == 0:
                        self.saver.save(self.sess, '%s/%04d/model.ckpt' % (results_dir, epoch))

    def test(self,
             dataset_dir, dataset_start_idx, dataset_end_idx,  # dataset
             in_channel,  # network input channel (rendered image + body-part masks)
             first_layer_channel,  # channel number of the first encoder layer
             results_dir='./test',  # results dir
             model_dir='./results512p'):  # dir of the model to load (only for 'resume' or 'finetune')

        with tf.name_scope('input'):
            with tf.variable_scope('input'):
                nin = tf.placeholder(tf.float32, [1, self.sp, self.sp, in_channel])
                real_image = tf.placeholder(tf.float32, [1, self.sp, self.sp, 3])
                lr = tf.placeholder(tf.float32)
        self._setup_model(first_layer_channel, nin, real_image)

        ckpt_prev = tf.train.get_checkpoint_state(model_dir)
        if ckpt_prev:
            saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()])
            saver.restore(self.sess, ckpt_prev.model_checkpoint_path)
            logger.write('Loaded model %s' % model_dir)
        else:
            logger.write('Unable to load the pretrained model. ')

        data_indices = range(dataset_start_idx, dataset_end_idx)
        logger.write('Initializing data loader. ')
        data_loader = DataLoader(dataset_dir, self.sp, self.sp, np.max(data_indices))

        cnt = 0
        for ind in data_indices:
            cnt += 1
            test_input_batch, test_target_batch = data_loader.get_data_pair(ind)
            output = self.sess.run(self.Gout_scaled, feed_dict={nin: test_input_batch, real_image: test_target_batch})
            scipy.misc.toimage(test_input_batch[0, :, :, 17:21], cmin=0, cmax=255).save(os.path.join(results_dir, 'input/input_%d.png' % ind))
            scipy.misc.toimage(test_target_batch[0, :, :, :], cmin=0, cmax=255).save(os.path.join(results_dir, 'gt/gt_%d.png' % ind))
            scipy.misc.toimage(output[0, :, :, :], cmin=0, cmax=255).save(os.path.join(results_dir, 'output/output_%d.png' % ind))
            if cnt % 10 == 0:
                logger.write('Processed %d images. ' % cnt)
        logger.write('In total, %d images were processed. The results are saved into %s. ' % (cnt, results_dir))

    def _setup_model(self, first_layer_ch, nin, real_image):
        logger.write('Setup network...')
        with tf.name_scope('generator'):
            with tf.variable_scope('G'):
                self.G = Generator(nin, 3, first_layer_ch, bottleneck_sp=8, bottleneck_ch=256, res_block_num=2)
                self.Gout = self.G.get_network_output()
                self.Gout_scaled = tf.scalar_mul(255.0, self.Gout)

        with tf.name_scope('fake_discriminator'):
            with tf.variable_scope('D'):
                self.D_fake = Discriminator(nin, self.Gout_scaled, first_layer_ch, disc_patch_res=self.discriminator_out_res)
                self.D_fake_out = self.D_fake.get_network_output()
        with tf.name_scope('real_discriminator'):
            with tf.variable_scope('D', reuse=True):
                self.D_real = Discriminator(nin, real_image, first_layer_ch, disc_patch_res=self.discriminator_out_res)
                self.D_real_out = self.D_real.get_network_output()

    def _setup_losses(self, real_image):
        logger.write('Setup losses...')
        assert self.G is not None
        assert self.D_fake is not None and self.D_real is not None

        with tf.name_scope('attention_map'):
            with tf.variable_scope('attention_map', reuse=tf.AUTO_REUSE):
                self.l1_diff_map = tf.reduce_sum(tf.abs(real_image - self.Gout_scaled), reduction_indices=[3], keepdims=True)
                self.l1_diff_map = tf.image.resize_bilinear(self.l1_diff_map,
                                                            (self.discriminator_out_res, self.discriminator_out_res),
                                                            align_corners=False)
                self.attention_map = self.l1_diff_map / tf.reduce_sum(self.l1_diff_map, reduction_indices=[1, 2], keepdims=True)

        with tf.name_scope('perception_loss'):
            with tf.variable_scope('perception_loss', reuse=tf.AUTO_REUSE):
                vgg_real = build_pretrained_vgg(real_image, '../VGG_Model/imagenet-vgg-verydeep-19.mat', reuse=False)
                vgg_fake = build_pretrained_vgg(self.Gout_scaled, '../VGG_Model/imagenet-vgg-verydeep-19.mat', reuse=True)

                lp_0 = compute_error(vgg_real['input'], vgg_fake['input'])
                lp_1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2']) * 5e-2
                lp_2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2']) * 5e-2
                lp_3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2']) * 5e-2
                lp_4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2']) * 5e-2
                lp_5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2']) * 5e-1
                lp = lp_0 + lp_1 + lp_2 + lp_3 + lp_4 + lp_5

        with tf.name_scope('disc_loss'):
            with tf.variable_scope('disc_loss', reuse=tf.AUTO_REUSE):
                g_disc_loss = tf.reduce_sum((tf.square(self.D_fake_out - 1)* self.attention_map))
                d_disc_loss = tf.reduce_sum((tf.square(self.D_fake_out)* self.attention_map)) + \
                    tf.reduce_sum((tf.square(self.D_real_out - 1)* self.attention_map))

        with tf.name_scope('loss'):
            with tf.variable_scope('loss'):
                self.G_content_loss = lp
                self.G_loss = self.G_content_loss + g_disc_loss * 0.25
                self.D_loss = d_disc_loss

    def _setup_optimizer(self, lr):
        logger.write('Setup optimizer...')
        assert self.D_loss is not None and self.G_loss is not None
        g_vars = [var for var in tf.trainable_variables() if var.name.startswith('G')]
        d_vars = [var for var in tf.trainable_variables() if var.name.startswith('D')]
        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.G_content_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.G_content_loss, var_list=g_vars)
                self.G_total_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.G_loss, var_list=g_vars)
                self.D_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.D_loss, var_list=d_vars)

    def _setup_summary(self, graph_dir, batch_size):
        logger.write('Setup summary...')
        # setup scalar summary
        s_gcl = tf.summary.scalar('loss_collection/G_content_loss', self.G_content_loss)
        s_gl = tf.summary.scalar('loss_collection/G_loss', self.G_loss)
        s_dl = tf.summary.scalar('loss_collection/D_loss', self.D_loss)
        self.merged_loss_scalar = tf.summary.merge([s_gcl, s_gl, s_dl])

        # setup image summary
        generator_output = tf.image.convert_image_dtype(self.Gout, dtype=tf.uint8, saturate=True)
        discriminator_output = tf.image.convert_image_dtype(self.D_fake_out, dtype=tf.uint8, saturate=True)
        l1_diff_map_output = tf.image.convert_image_dtype(self.l1_diff_map / 255, dtype=tf.uint8, saturate=True)
        s_ti_g = tf.summary.image('testing_image_g', generator_output, max_outputs=batch_size)
        s_ti_d = tf.summary.image('testing_image_d', discriminator_output, max_outputs=batch_size)
        s_ti_am = tf.summary.image('testing_image_am', l1_diff_map_output, max_outputs=batch_size)
        self.merged_testing = tf.summary.merge([s_ti_g, s_ti_d, s_ti_am])

        self.writer = tf.summary.FileWriter(graph_dir, self.sess.graph)
