# coding=utf-8
import os
import torch
from training.datasets.coco_data.heatmap import putGaussianMaps
from training.datasets.coco_data.ImageAugmentation import (aug_croppad, aug_flip,
                                                  aug_rotate, aug_scale)
from training.datasets.coco_data.paf import putVecMaps
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess, dense_pose_preprocess)
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
 id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee,
     5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top,
     10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder,
     14 - l elbow, 15 - l wrist)
'''

class Cocokeypoints(Dataset):
    def __init__(self, root, mask_dir, index_list, data, inp_size, feat_stride, preprocess='rtpose', transform=None,
                 target_transform=None, params_transform=None):

        self.params_transform = params_transform
        self.params_transform['crop_size_x'] = inp_size
        self.params_transform['crop_size_y'] = inp_size
        self.params_transform['stride'] = feat_stride

        # add preprocessing as a choice, so we don't modify it manually.
        self.preprocess = preprocess
        self.data = data
        self.mask_dir = mask_dir
        self.numSample = len(index_list)
        self.index_list = index_list
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def get_anno(self, meta_data):
        """
        get meta information
        """
        anno = dict()
        anno['dataset'] = meta_data['dataset']
        anno['img_height'] = int(meta_data['img_height'])
        anno['img_width'] = int(meta_data['img_width'])

        anno['isValidation'] = meta_data['isValidation']
        anno['people_index'] = int(meta_data['people_index'])
        anno['annolist_index'] = int(meta_data['annolist_index'])

        # (b) objpos_x (float), objpos_y (float)
        anno['objpos'] = np.array(meta_data['objpos'])
        anno['scale_provided'] = meta_data['scale_provided']
        anno['joint_self'] = np.array(meta_data['joint_self'])

        anno['numOtherPeople'] = int(meta_data['numOtherPeople'])
        if anno['numOtherPeople'] > 1:
            # print(np.array(anno['joint_others']).shape)
            if len(np.array(meta_data['joint_others']).shape) != 1:
                anno['joint_others'] = np.array(meta_data['joint_others'])
            else:
                n = int(anno['numOtherPeople'])
                tempa = np.zeros((n, 16, 3))
                for a in range(int(anno['numOtherPeople'])):
                    for b in range(len(meta_data['joint_others'][a])):
                        tempa[a, b, :] = meta_data['joint_others'][a][b]
                anno['joint_others'] = tempa
        anno['objpos_other'] = np.array(meta_data['objpos_other'])
        anno['scale_provided_other'] = meta_data['scale_provided_other']
        if anno['numOtherPeople'] == 1:
            anno['joint_others'] = np.array(meta_data['joint_others'])
            anno['joint_others'] = np.expand_dims(anno['joint_others'], 0)
            anno['objpos_other'] = np.expand_dims(anno['objpos_other'], 0)
        return anno


    def remove_illegal_joint(self, meta):
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])
        mask = np.logical_or.reduce((meta['joint_self'][:, 0] >= crop_x,
                                     meta['joint_self'][:, 0] < 0,  # fix mpii json
                                     meta['joint_self'][:, 1] >= crop_y,
                                     meta['joint_self'][:, 1] < 0)) # fix mpii json
        # out_bound = np.nonzero(mask)
        # print(mask.shape)
        meta['joint_self'][mask == True, :] = (1, 1, 2)
        if (meta['numOtherPeople'] != 0):
            mask = np.logical_or.reduce((meta['joint_others'][:, :, 0] >= crop_x,
                                         meta['joint_others'][:, :, 0] < 0,  # fix mpii json
                                         meta['joint_others'][:,
                                                              :, 1] >= crop_y,
                                         meta['joint_others'][:, :, 1] < 0))  # fix mpii json <=
            meta['joint_others'][mask == True, :] = (1, 1, 2)

        return meta

    def get_ground_truth(self, meta, mask_miss):

        stride = self.params_transform['stride']
        crop_size_y = self.params_transform['crop_size_y']
        crop_size_x = self.params_transform['crop_size_x']
        nop = meta['numOtherPeople']
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        heatmaps = np.zeros((int(grid_y), int(grid_x), 16))
        pafs = np.zeros((int(grid_y), int(grid_x), 32))

        mask_miss = cv2.resize(mask_miss, (0, 0), fx=1.0 / stride, fy=1.0 /
                               stride, interpolation=cv2.INTER_CUBIC).astype(
            np.float32)
        mask_miss = mask_miss / 255.
        mask_miss = np.expand_dims(mask_miss, axis=2)

        heat_mask = np.repeat(mask_miss, 16, axis=2)
        paf_mask = np.repeat(mask_miss, 32, axis=2)
        '''
         id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee,
             5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top,
             10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder,
             14 - l elbow, 15 - l wrist)
        '''
        # now 6 is the center of person, and thorax was eliminated.
        ran = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        for i in range(15):
            # the first person'
            index = ran[i]
            if (meta['joint_self'][index, 2] <= 1):
                center = meta['joint_self'][index, :2]
                gaussian_map = heatmaps[:, :, i]
                heatmaps[:, :, i] = putGaussianMaps(
                    center, gaussian_map, params_transform=self.params_transform)
            # the other people in the image
            for j in range(nop):
                if (meta['joint_others'][j, index, 2] <= 1):
                    center = meta['joint_others'][j, index, :2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(
                        center, gaussian_map, params_transform=self.params_transform)

        mid_1 = [8, 8, 6, 3, 4, 6, 2, 1, 8,
                 13, 14, 8, 12, 11, 12, 13]

        mid_2 = [9, 6, 3, 4, 5, 2, 1, 0, 13,
                 14, 15, 12, 11, 10, 9, 9]

        for i in range(16):
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            if (meta['joint_self'][mid_1[i], 2] <= 1 and meta['joint_self'][mid_2[i], 2] <= 1):
                centerA = meta['joint_self'][mid_1[i], :2]
                centerB = meta['joint_self'][mid_2[i], :2]
                vec_map = pafs[:, :, 2 * i:2 * i + 2]
                pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                                centerB=centerB,
                                                                accumulate_vec_map=vec_map,
                                                                count=count, params_transform=self.params_transform)
            for j in range(nop):
                if (meta['joint_others'][j, mid_1[i], 2] <= 1 and meta['joint_others'][j, mid_2[i], 2] <= 1):
                    centerA = meta['joint_others'][j, mid_1[i], :2]
                    centerB = meta['joint_others'][j, mid_2[i], :2]
                    vec_map = pafs[:, :, 2 * i:2 * i + 2]
                    pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                                    centerB=centerB,
                                                                    accumulate_vec_map=vec_map,
                                                                    count=count, params_transform=self.params_transform)
        # background
        heatmaps[:, :, -1] = np.maximum(1 - np.max(heatmaps[:, :, :14], axis=2), 0.)

        return heat_mask, heatmaps, paf_mask, pafs

    def remove_zero(self, meta):

        nop = meta['numOtherPeople']
        for i in range(16):
            if (meta['joint_self'][i, 0] == 0 and meta['joint_self'][i, 1] == 0):
                meta['joint_self'][i, 2] = 2

            for j in range(nop):
                if (meta['joint_others'][j, i, 0] == 0 and meta['joint_others'][j, i, 1] == 0):
                    meta['joint_others'][j, i, 2] = 2

        return meta

    ### add center of person
    def get_anno_inmyorder(self, meta):
        nop = meta['numOtherPeople']
        ox, oy = (meta['joint_self'][6, :2] + meta['joint_self'][7, :2]) / 2
        meta['joint_self'][6, 0] = ox
        meta['joint_self'][6, 1] = oy

        for j in range(nop):
            ox, oy = (meta['joint_others'][j, 6, :2] + meta['joint_others'][j, 7, :2])/2
            meta['joint_others'][j, 6, 0] = ox
            meta['joint_others'][j, 6, 1] = oy

        return meta

    def __getitem__(self, index):

        idx = self.index_list[index]
        img = cv2.imread(os.path.join(self.root, self.data[idx]['img_paths']))
        img_idx = self.data[idx]['img_paths'][:-3]
        mask_miss = cv2.imread(self.mask_dir + '/masks/mask_' + img_idx + 'jpg', 0)
        # fix mask problems
        mask_miss[mask_miss > 0] = 200.
        mask_miss[mask_miss == 0] = 255.
        mask_miss[mask_miss == 200] = 0.

        meta_data = self.get_anno(self.data[idx])
        meta_data = self.remove_zero(meta_data)
        meta_data = self.get_anno_inmyorder(meta_data)
        meta_data, img, mask_miss = aug_scale(
            meta_data, img, mask_miss, self.params_transform)

        meta_data, img, mask_miss = aug_rotate(
            meta_data, img, mask_miss, self.params_transform)
        meta_data, img, mask_miss = aug_croppad(
            meta_data, img, mask_miss, self.params_transform)
        meta_data, img, mask_miss = aug_flip(
            meta_data, img, mask_miss, self.params_transform, coco=False, neworder=True)
        meta_data = self.remove_illegal_joint(meta_data)
        heat_mask, heatmaps, paf_mask, pafs = self.get_ground_truth(
            meta_data, mask_miss)

        ##############################################
        ##########      check labels      ############
        ##############################################
        # mask = cv2.resize(mask_miss, (384, 384))
        # image = img
        # heatmaps = cv2.resize(heatmaps, (384, 384))
        # pafs = cv2.resize(pafs, (384, 384))
        # mask = mask / 255
        # mask = mask.astype(np.uint8)
        # mask = np.expand_dims(mask, axis=2)
        # mask = np.repeat(mask, 3, axis=2)
        # image = cv2.multiply(mask, image)
        # for j in range(0, 16, 4):
        #     heatmap = heatmaps[:, :, j]
        #     heatmap = heatmap.reshape((384, 384, 1))
        #     heatmap *= 255
        #     heatmap = heatmap.astype(np.uint8)
        #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     plt.imshow(image)
        #     plt.imshow(heatmap, alpha=0.5)
        #     plt.show()
        #     plt.close()
        # for j in range(0, 32, 2):
        #     paf = np.abs(pafs[:, :, j])
        #     paf += np.abs(pafs[:, :, j + 1])
        #     paf[paf > 1] = 1
        #     paf = paf.reshape((384, 384, 1))
        #     paf *= 255
        #     paf = paf.astype(np.uint8)
        #     paf = cv2.applyColorMap(paf, cv2.COLORMAP_JET)
        #     plt.imshow(image)
        #     plt.imshow(paf, alpha=0.5)
        #     plt.show()
        #     plt.close()

        ##############################################
        ##########      check labels      ############
        ##############################################

        # trianed on Imagenet dataset
        if self.preprocess == 'rtpose':
            img = rtpose_preprocess(img)

        elif self.preprocess == 'vgg':
            img = vgg_preprocess(img)

        elif self.preprocess == 'inception':
            img = inception_preprocess(img)

        elif self.preprocess == 'ssd':
            img = ssd_preprocess(img)

        elif self.preprocess == 'atrous_pose':
            img = dense_pose_preprocess(img)

        img = torch.from_numpy(img)
        heatmaps = torch.from_numpy(
            heatmaps.transpose((2, 0, 1)).astype(np.float32))
        heat_mask = torch.from_numpy(
            heat_mask.transpose((2, 0, 1)).astype(np.float32))
        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))
        paf_mask = torch.from_numpy(
            paf_mask.transpose((2, 0, 1)).astype(np.float32))

        return img, heatmaps, heat_mask, pafs, paf_mask, self.data[idx]['img_paths'], idx

    def __len__(self):
        return self.numSample
