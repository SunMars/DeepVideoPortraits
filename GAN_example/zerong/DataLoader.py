from __future__ import division, print_function, absolute_import
import numpy as np
import scipy
import scipy.io
import psutil


class DataLoader(object):
    def __init__(self, datadir, input_size, target_size, data_max_ind):
        self.datadir = datadir
        self.input_size = input_size
        self.target_size = target_size
        self.cut_left = int((1920 - 1024) / 2)
        self.cut_right = int(1920 - self.cut_left)
        self.cut_top = int((1080 - 1024) / 2)
        self.cut_down = int(1080 - self.cut_top)

        # read background image
        self.bg = scipy.misc.imread(datadir + '/bg.png')
        self.bg = self._crop_img(self.bg)
        self.bg = scipy.misc.imresize(self.bg, (self.target_size, self.target_size))

        # init buffer
        self.rgb_buf = [None] * (data_max_ind+1)
        self.texture_buf = [None] * (data_max_ind+1)
        self.bodypart_buf = [None] * (data_max_ind+1)

    def preload_data(self, indices, max_mem_percent):
        print('-- Pre-loading some data into memory. ')
        cnt = 0
        for ind in indices:
            self.rgb_buf[ind] = self._load_real_rgb_image(ind, self.target_size)
            self.texture_buf[ind] = self._load_texture_img(ind, self.input_size)
            self.bodypart_buf[ind] = self._load_bodypart_mask_img(ind, self.input_size)

            cnt += 1
            if cnt % 10 == 0:
                info = psutil.virtual_memory()
                print('-- Loaded %d data pairs. Current memory usage: %.1f%%' % (cnt, info.percent))
                if info.percent > max_mem_percent:
                    break

        print('-- In total, %d data pairs are loaded into memory. ' % cnt)

    def get_data_pair(self, id):
        mask = self.get_bodypart_img(id)
        render = self.get_texture_img(id)
        rgb = self.get_real_rgb_img(id)
        d_in = np.expand_dims(np.concatenate((mask, render, self.bg), axis=-1), axis=0)
        d_tar = np.expand_dims(rgb, axis=0)
        return d_in, d_tar

    def get_bg(self):
        return self.bg

    def get_texture_img(self, id):
        if self.texture_buf[id] is None:
            return self._load_texture_img(id, self.input_size)
        else:
            return self.texture_buf[id]

    def get_real_rgb_img(self, id):
        if self.rgb_buf[id] is None:
            return self._load_real_rgb_image(id, self.target_size)
        else:
            return self.rgb_buf[id]

    def get_bodypart_img(self, id):
        if self.bodypart_buf[id] is None:
            return self._load_bodypart_mask_img(id, self.input_size)
        else:
            return self.bodypart_buf[id]

    def _load_texture_img(self, id, img_size):
        fname = '%s/pose-%d-render.png' % (self.datadir+'/render', id)
        img = scipy.misc.imread(fname)
        img = self._crop_img(img)
        img = scipy.misc.imresize(img, (img_size, img_size))
        return np.float32(img)

    def _load_bodypart_mask_img(self, id, img_size):
        bp_names = ['0', '1', '2', '4', '5', '3_6', '7_10', '8_11', '9_13_14', '12_15', '16', '17', '18', '19', '20_22',
                    '21_23']
        masks = [None] * (len(bp_names) + 1)
        for bpid, bp_name in enumerate(bp_names):
            fname = '%s/pose-%d-bodypart_%s.png' % (self.datadir+'/render', id, bp_name)
            img = scipy.misc.imread(fname)
            img = self._crop_img(img)
            img = scipy.misc.imresize(img, (img_size, img_size))
            masks[bpid] = np.expand_dims(np.float32(img), axis=-1)

        masks[-1] = 1 - np.sum(np.concatenate(tuple(masks[0:len(bp_names)]), axis=-1), axis=-1)
        masks[-1] = np.expand_dims(masks[-1], axis=-1)
        concat_masks = np.concatenate(tuple([mask for mask in masks]), axis=-1)
        return concat_masks

    def _load_real_rgb_image(self, id, img_size):
        fname = '%s/frame_%d.png' % (self.datadir+'/fullcolor', id)
        img = scipy.misc.imread(fname)
        img = self._crop_img(img)
        img = scipy.misc.imresize(img, (img_size, img_size))
        return np.float32(img)

    def _crop_img(self, img):
        if len(img.shape) == 3:
            return img[self.cut_top:self.cut_down, self.cut_left:self.cut_right, :]
        elif len(img.shape) == 2:
            return img[self.cut_top:self.cut_down, self.cut_left:self.cut_right]


