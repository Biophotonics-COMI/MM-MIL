"""
Evaluation functions for SLAM

Copyright (c) 2020 University of Illinois at Urbana-Champaign
(see LICENSE for details)
Written by Jindou Shi
"""

import os
import sys
import cv2
import json
import time
import random
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from scipy.stats import rankdata

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from . import stainSLAM as ss
from . import model as mil_model
from . import utils as mil_utils
from . import dataset as mil_data


class SLAMslide():

    def __init__(self, basename, tile_size, grid_coords, grid_probs, label, pred, saliency_maps=None,
                 prob_thres=0.90, bright_adj=[2,4,0.5,0.8]):

        self.basename = basename
        # self.channel_paths = channel_paths
        self.tile_size = tile_size
        self.grid_coords = grid_coords  # [[x1,y1], [x2, y2], ...]

        self.grid_probs = list(grid_probs)  # np.array([prob1, prob2, ...])
        self.prob_thres = prob_thres
        self.bright_adj = bright_adj
        self.label = label
        self.pred = pred
        grid_pos_coords = [coord for i, coord in enumerate(self.grid_coords) if self.grid_probs[i] >= self.prob_thres]
        self.grid_pos_coords = grid_pos_coords  # [[x2,y2], [x8, y8], ...]
        self.grid_pos_ids = [i for i, coord in enumerate(self.grid_coords) if self.grid_probs[i] >= self.prob_thres]
        self.saliency_maps = saliency_maps
        print(len(self.grid_pos_ids), 'positive tiles')

    '''def wsi_raw(self):
        im_2 = cv2.imread(self.path_2pf, cv2.IMREAD_GRAYSCALE) / 255.
        im_3 = cv2.imread(self.path_3pf, cv2.IMREAD_GRAYSCALE) / 255.
        im_s = cv2.imread(self.path_shg, cv2.IMREAD_GRAYSCALE) / 255.
        im_t = cv2.imread(self.path_thg, cv2.IMREAD_GRAYSCALE) / 255.
        wsi = ss.stain_SLAM(im_2, im_3, im_s, im_t, self.bright_adj)  # shape:
        return wsi'''

    def wsi_read(self, wsi_path):
        wsi = cv2.imread(wsi_path, cv2.IMREAD_COLOR)
        wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)
        return wsi

    def wsi_prob_array(self, wsi):
        xpix, ypix, _ = wsi.shape
        prob_array = np.zeros((xpix, ypix))
        length = self.tile_size
        for i, tile in enumerate(self.grid_coords):
            x, _, y, _ = tile
            prob = self.grid_probs[i]
            temp_array = np.zeros((xpix, ypix))
            temp_array[x:x+length, y:y+length] = prob
            temp_flag = temp_array > prob_array
            prob_array[temp_flag] = prob
        return prob_array

    def tile_saliency_map(self, wsi_path=None, save_folder=None):
        wsi = self.wsi_read(wsi_path)

        '''if wsi_path is not None:
            wsi = self.wsi_read(wsi_path)
        else:
            wsi = self.wsi_raw()'''

        for coords, pos_id in zip(self.grid_pos_coords, self.grid_pos_ids):
            # print(tile)
            x1, x2, y1, y2 = coords
            tile_img = wsi[x1:x2, y1:y2, :]
            s_maps = np.float32(self.saliency_maps[pos_id, :, :, :])
            new_smap = np.sum(s_maps, axis=0)

            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(tile_img)
            ax.set_title('P = %.3f'%self.grid_probs[pos_id])
            ax.axis('off')
            save_name = self.basename + '_tile{}.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(new_smap)
            ax.set_title('Combined')
            ax.axis('off')
            save_name = self.basename + '_tile{}_saliency_all.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name))
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(s_maps[0, :, :])
            ax.set_title('SHG')
            ax.axis('off')
            save_name = self.basename + '_tile{}_saliency_SHG.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(s_maps[1, :, :])
            ax.set_title('THG')
            ax.axis('off')
            save_name = self.basename + '_tile{}_saliency_THG.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(s_maps[2, :, :])
            ax.set_title('2PF')
            ax.axis('off')
            save_name = self.basename + '_tile{}_saliency_2PF.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(s_maps[3, :, :])
            ax.set_title('3PF')
            ax.axis('off')
            save_name = self.basename + '_tile{}_saliency_3PF.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

    def tile_smoothgrad(self, wsi_path=None, save_folder=None):
        wsi = self.wsi_read(wsi_path)

        '''if wsi_path is not None:
            wsi = self.wsi_read(wsi_path)
        else:
            wsi = self.wsi_raw()'''

        for coords, pos_id in zip(self.grid_pos_coords, self.grid_pos_ids):
            # print(tile)
            x1, x2, y1, y2 = coords
            tile_img = wsi[x1:x2, y1:y2, :]
            s_maps = np.float32(self.saliency_maps[pos_id, :, :, :])
            new_smap = np.sum(s_maps, axis=0)

            # gray original image + saliency map
            cmap_name = 'B2R'
            colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]
            n_bin = 200
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
            gray = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
            value_max = np.max(np.absolute(new_smap))
            figure, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(gray, cmap='gray')
            ax.imshow(new_smap, cmap=cm, alpha=.5, vmin=-1. * value_max, vmax=value_max)
            ax.set_title('P = %.3f' % self.grid_probs[pos_id])
            ax.axis('off')
            save_name = self.basename + '_tile{}_overlay.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            # gray original image + saliency map percentile, absolute value
            cmap_name = 'black-red'
            colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0)]
            n_bin = 200
            cm2 = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

            # process smap: 1. absolute value; 2. match to percentile
            abs_smap = np.absolute(new_smap)
            abs_smap_rank = rankdata(abs_smap) - 1
            value_max = np.max(abs_smap_rank)
            abs_map_percentile = abs_smap_rank / value_max
            abs_map_percentile = abs_map_percentile.reshape(abs_smap.shape[0], abs_smap.shape[1])

            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(gray, cmap='gray')
            ax.imshow(abs_map_percentile, cmap=cm2, alpha=.5)
            ax.set_title('Percentile saliency map')
            ax.axis('off')
            save_name = self.basename + '_tile{}_overlay_percentile.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            '''figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(new_smap)
            ax.set_title('Combined')
            ax.axis('off')
            save_name = self.basename + '_tile{}_SmoothGrad_all.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name))
            plt.close()'''

            '''figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = new_smap
            value_max = np.max(np.absolute(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1. * value_max, vmax=value_max)
            ax.set_title('SaliencyMap_Combined')
            # ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_all.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            all_max_gradients = []
            all_max_positive = []
            all_max_negative = []

            figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = s_maps[0, :, :]
            value_max = np.max(np.absolute(temp_img))
            all_max_gradients.append(value_max)
            all_max_positive.append(np.max(temp_img))
            all_max_negative.append(np.min(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1. * value_max, vmax=value_max)
            ax.set_title('SHG')
            #ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_SHG.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = s_maps[1, :, :]
            value_max = np.max(np.absolute(temp_img))
            all_max_gradients.append(value_max)
            all_max_positive.append(np.max(temp_img))
            all_max_negative.append(np.min(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1. * value_max, vmax=value_max)
            ax.set_title('THG')
            #ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_THG.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = s_maps[2, :, :]
            value_max = np.max(np.absolute(temp_img))
            all_max_gradients.append(value_max)
            all_max_positive.append(np.max(temp_img))
            all_max_negative.append(np.min(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1. * value_max, vmax=value_max)
            ax.set_title('2PF')
            #ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_2PF.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = s_maps[3, :, :]
            value_max = np.max(np.absolute(temp_img))
            all_max_gradients.append(value_max)
            all_max_positive.append(np.max(temp_img))
            all_max_negative.append(np.min(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1.*value_max, vmax=value_max)
            ax.set_title('3PF')
            #ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_3PF.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            # channel contribution comparison (bar chart for max gradients of each channel)
            y_pos = np.arange(len(all_max_gradients))
            objects = ['SHG', 'THG', '2PF', '3PF']
            plt.figure(figsize=(5, 5))
            plt.bar(y_pos, all_max_gradients, align='center', color=['g', 'm', 'y', 'c'])
            plt.xticks(y_pos, objects)
            plt.ylabel('Maximum absolute gradient')
            plt.title('Channel Importance')
            save_name = self.basename + '_tile{}_barchart.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            # channel contribution comparison (both positive and negative)
            y_pos = np.arange(len(all_max_gradients))
            objects = ['SHG', 'THG', '2PF', '3PF']
            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.bar(y_pos, all_max_positive, align='center', color=['g', 'm', 'y', 'c'])
            ax.bar(y_pos, all_max_negative, align='center', color=['g', 'm', 'y', 'c'])
            plt.xticks(y_pos, objects)
            plt.ylabel('Channel gradient')
            plt.title('Channel Attribution')
            save_name = self.basename + '_tile{}_barchart2.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()'''

    def tile_smoothgrad_original(self, wsi_path=None, save_folder=None):
        wsi = self.wsi_read(wsi_path)

        '''if wsi_path is not None:
            wsi = self.wsi_read(wsi_path)
        else:
            wsi = self.wsi_raw()'''

        for coords, pos_id in zip(self.grid_pos_coords, self.grid_pos_ids):
            # print(tile)
            x1, x2, y1, y2 = coords
            tile_img = wsi[x1:x2, y1:y2, :]
            s_maps = np.float32(self.saliency_maps[pos_id, :, :, :])
            new_smap = np.sum(s_maps, axis=0)

            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(tile_img)
            ax.set_title('P = %.3f'%self.grid_probs[pos_id])
            ax.axis('off')
            save_name = self.basename + '_tile{}.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            '''figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.imshow(new_smap)
            ax.set_title('Combined')
            ax.axis('off')
            save_name = self.basename + '_tile{}_SmoothGrad_all.png'.format(pos_id)
            plt.savefig(os.path.join(save_folder, save_name))
            plt.close()'''

            figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = new_smap
            value_max = np.max(np.absolute(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1. * value_max, vmax=value_max)
            ax.set_title('SaliencyMap_Combined')
            # ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_all.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            all_max_gradients = []
            all_max_positive = []
            all_max_negative = []

            figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = s_maps[0, :, :]
            value_max = np.max(np.absolute(temp_img))
            all_max_gradients.append(value_max)
            all_max_positive.append(np.max(temp_img))
            all_max_negative.append(np.min(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1. * value_max, vmax=value_max)
            ax.set_title('SHG')
            ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_SHG.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = s_maps[1, :, :]
            value_max = np.max(np.absolute(temp_img))
            all_max_gradients.append(value_max)
            all_max_positive.append(np.max(temp_img))
            all_max_negative.append(np.min(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1. * value_max, vmax=value_max)
            ax.set_title('THG')
            ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_THG.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = s_maps[2, :, :]
            value_max = np.max(np.absolute(temp_img))
            all_max_gradients.append(value_max)
            all_max_positive.append(np.max(temp_img))
            all_max_negative.append(np.min(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1. * value_max, vmax=value_max)
            ax.set_title('2PF')
            ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_2PF.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            figure, ax = plt.subplots(1, figsize=(5, 5))
            temp_img = s_maps[3, :, :]
            value_max = np.max(np.absolute(temp_img))
            all_max_gradients.append(value_max)
            all_max_positive.append(np.max(temp_img))
            all_max_negative.append(np.min(temp_img))
            im = ax.imshow(temp_img, cmap='seismic', vmin=-1.*value_max, vmax=value_max)
            ax.set_title('3PF')
            ax.axis('off')
            # figure.colorbar(im, ax=ax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_name = self.basename + '_tile{}_SmoothGrad_3PF.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            # channel contribution comparison (bar chart for max gradients of each channel)
            y_pos = np.arange(len(all_max_gradients))
            objects = ['SHG', 'THG', '2PF', '3PF']
            plt.figure(figsize=(5, 5))
            plt.bar(y_pos, all_max_gradients, align='center', color=['g', 'm', 'y', 'c'])
            plt.xticks(y_pos, objects)
            plt.ylabel('Maximum absolute gradient')
            plt.title('Channel Importance')
            save_name = self.basename + '_tile{}_barchart.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

            # channel contribution comparison (both positive and negative)
            y_pos = np.arange(len(all_max_gradients))
            objects = ['SHG', 'THG', '2PF', '3PF']
            figure, ax = plt.subplots(1, figsize=(5, 5))
            ax.bar(y_pos, all_max_positive, align='center', color=['g', 'm', 'y', 'c'])
            ax.bar(y_pos, all_max_negative, align='center', color=['g', 'm', 'y', 'c'])
            plt.xticks(y_pos, objects)
            plt.ylabel('Channel gradient')
            plt.title('Channel Attribution')
            save_name = self.basename + '_tile{}_barchart2.png'.format(pos_id)
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
            plt.close()

    def wsi_overlay(self, wsi_path=None, save_folder=None):
        wsi = self.wsi_read(wsi_path)

        '''if wsi_path is not None:
            wsi = self.wsi_read(wsi_path)
        else:
            wsi = self.wsi_raw()'''

        figure, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(wsi)
        # ax.set_title('{} (ground truth: {}, prediction: {})'.format(self.basename, self.label, self.pred))
        ax.set_title(self.basename)
        ax.axis('off')
        length = self.tile_size

        for tile in self.grid_pos_coords:
            # print(tile)
            x, _, y, _ = tile
            rect = patches.Rectangle((y, x), length, length, edgecolor='r', ls=':',
                                     linewidth=1., facecolor="r", alpha=0.3)
            ax.add_patch(rect)
        plt.axis('off')
        if save_folder is None:
            plt.show()
        else:
            save_name = self.basename + '_whole-slide-overlay.png'
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
        plt.close()

    def wsi_prob_map(self, wsi_path=None, save_folder=None):
        wsi = self.wsi_read(wsi_path)

        '''if wsi_path is not None:
            wsi = self.wsi_read(wsi_path)
        else:
            wsi = self.wsi_raw()'''

        # colormap
        cmap_name = 'B2R'
        # colors = [(1, 0, 0), (0.6, 0, 0), (0, 0, 0.6), (0, 0, 1)]
        colors = [(0, 0, 1), (0, 0, 0.6), (0.6, 0, 0), (1, 0, 0)]
        n_bin = 200
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

        figure, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(wsi)
        # ax.set_title('{} (ground truth: {}, prediction: {})'.format(self.basename, self.label, self.pred))
        ax.set_title(self.basename)
        ax.axis('off')

        prob_array = self.wsi_prob_array(wsi)
        ax.imshow(prob_array, cmap=cm, alpha=0.5, vmax=1., vmin=0.)
        ax.axis('off')
        plt.axis('off')
        if save_folder is None:
            plt.show()
        else:
            save_name = self.basename + '_wsi-prob-map.png'
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        im = ax.imshow(prob_array, cmap=cm, vmin=0.0, vmax=1.0)
        # ax.set_title('{} (ground truth: {}, prediction: {})'.format(self.basename, self.label, self.pred))
        ax.set_title(self.basename)
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        '''
        figure, ax = plt.subplots(1, figsize=(15, 15))
        im = ax.imshow(prob_array, cmap=cm, vmin=0.0, vmax=1.0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('{}'.format(self.basename))
        ax.axis('off')
        figure.colorbar(im, cax=cax)
        # cbar = figure.colorbar(im, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # cbar.ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        '''

        save_name = self.basename + '_colormap.png'
        plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
        plt.close()

    def wsi_tsne_rgb(self, tsne_array, wsi_path=None, save_folder=None):
        wsi = self.wsi_read(wsi_path)

        '''if wsi_path is not None:
            wsi = self.wsi_read(wsi_path)
        else:
            wsi = self.wsi_raw()'''

        # generate tsne rgb overlay
        xpix, ypix, _ = wsi.shape
        shadow_flag = np.zeros((xpix, ypix, 3), dtype=np.uint8)
        shadow_wsi = np.zeros((xpix, ypix, 3), dtype=np.uint8)

        length = self.tile_size
        for i, tile in enumerate(self.grid_coords):
            x, _, y, _ = tile
            color = tsne_array[i, :].copy()
            color = np.uint8(color*255)   # shape (3,)

            '''
            # average
            temp_s_flag = shadow_flag[x:x+length, y:y+length, :].copy()
            temp_s_tile = shadow_wsi[x:x+length, y:y+length, :].copy()
            temp_s_flag = temp_s_flag + 1
            temp_s_tile = temp_s_tile + color.reshape(1, 1, 3)
            temp_s_tile = temp_s_tile / temp_s_flag
            shadow_wsi[x:x+length, y:y+length, :] = temp_s_tile.copy()
            shadow_flag[x:x + length, y:y + length, :] = 1
            '''
            # newest
            shadow_wsi[x:x + length, y:y + length, :] = color.reshape(1, 1, 3)

        figure, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(wsi)
        ax.set_title(self.basename)
        ax.axis('off')
        ax.imshow(shadow_wsi, alpha=0.5)
        ax.axis('off')
        plt.axis('off')
        if save_folder is None:
            plt.show()
        else:
            save_name = self.basename + '_tsne-overlay.png'
            plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.imshow(shadow_wsi)
        ax.set_title(self.basename)
        ax.axis('off')
        save_name = self.basename + '_tsne_colormap.png'
        plt.savefig(os.path.join(save_folder, save_name), dpi=figure.dpi)
        plt.close()




















'''
class Patient():
    def __init__(self):
        a = 1
'''


