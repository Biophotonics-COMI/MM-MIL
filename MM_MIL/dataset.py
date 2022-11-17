"""
Datasets functions

Lite version
save tile images to a folder
ignore preprocessing and tiling step

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
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tif
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torch.backends.cudnn as cudnn1
import torchvision.transforms as transforms


def generate_grid(xpix, ypix, tile_size, overlap):
    # only one area (for test dataset)
    all_coords = []
    count_x = int((xpix - tile_size) // (tile_size * (1 - overlap)) + 1)
    count_y = int((ypix - tile_size) // (tile_size * (1 - overlap)) + 1)

    tile_id = 0

    for i in range(count_x):
        for j in range(count_y):
            x = int(i * (tile_size * (1 - overlap)))
            y = int(j * (tile_size * (1 - overlap)))
            all_coords.append([x, x+tile_size, y, y+tile_size])
            tile_id += 1
    return all_coords


def generate_grid_4area(xpix, ypix, tile_size, overlap):

    count_x = int((xpix - tile_size) // (tile_size * (1 - overlap)) + 1)
    count_y = int((ypix - tile_size) // (tile_size * (1 - overlap)) + 1)

    all_coords = [[], [], [], []]

    tile_id = 0

    for i in range(count_x):
        for j in range(count_y):
            x = int(i * (tile_size * (1 - overlap)))
            y = int(j * (tile_size * (1 - overlap)))

            # new 2021 1016 check which area this tile belongs to
            center_x = x + tile_size / 2
            center_y = y + tile_size / 2

            if center_x < xpix / 2 and center_y < ypix / 2:
                all_coords[0].append([x, x + tile_size, y, y + tile_size])
            elif center_x >= xpix / 2 and center_y < ypix / 2:
                all_coords[1].append([x, x + tile_size, y, y + tile_size])
            elif center_x < xpix / 2 and center_y >= ypix / 2:
                all_coords[2].append([x, x + tile_size, y, y + tile_size])
            else:
                all_coords[3].append([x, x + tile_size, y, y + tile_size])
            tile_id += 1
    return all_coords


def write_tiff_file(path, img_content, swap=True):
    """
    Write a numpy array to file (.tiff 32-bit, 0.0~1.0)
    :param path: (str) the output folder
    :param img_content: (ndarray) a numpy array (xpix, ypix, tpix)
    :param swap: (bool) By default, swap the axis of the image cube to (tpix, xpix, ypix)
    :return:
        0: success
        1: failed
    """
    if swap:
        img_content = np.swapaxes(img_content, 0, 2)
        img_content = np.swapaxes(img_content, 1, 2)
    img_content = np.float32(img_content)
    output_file = os.path.abspath(path)
    imageio.mimwrite(output_file, img_content)
    return 0


def read_tiff_file(file_path):
    try:
        im = tif.imread(file_path)
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 0, 1)
    except:
        print("Error: File does not appear to exist.\nFile path: " + file_path)
        return None
    return im


def read_tiff_file(file_path, swap=False):
    """
    Reads a tiff file and return a data cube
    :param file_path: (str) path of the tiff file
    :param swap: (bool) by default, the shape of img_content is (xpix, ypix, tpix); if False, (tpix, xpix, ypix)
    :return:
        img_content: (ndarray) a image cube
    """

    try:
        tiff_file = Image.open(file_path)
    except:
        print("Error: File does not appear to exist.\nFile path: " + file_path)
        return [], {}
    tpix = tiff_file.n_frames
    img_content = []
    for frame_id in range(tpix):
        tiff_file.seek(frame_id)
        img_content.append(np.asarray(tiff_file))
    img_content = np.asarray(img_content)
    if swap:
        img_content = np.swapaxes(img_content, 0, 2)
        img_content = np.swapaxes(img_content, 0, 1)
    img_content = np.squeeze(img_content)
    return img_content


def img_transform(img, img_trans):
    if img_trans == 'original':
        img_out = img
    elif img_trans == 'v_flip':
        img_out = cv2.flip(img, 0)
    elif img_trans == 'h_flip':
        img_out = cv2.flip(img, 1)
    elif img_trans == 'vh_flip':
        img_out = cv2.flip(img, -1)
    elif img_trans == 'rot+90':
        img_out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif img_trans == 'rot-90':
        img_out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_out


def make_dataset_train_val(indexfile, config, role, save_folder, grid_folder):
    # make the whole dataset (without data augmentation, except z-score normalization)
    # !!! update 2021 1016: new bag policy (each image will be split into 4 areas)

    slide_basename = config["dataset"]["images"]["image_names"]
    channels = config["dataset"]["images"]["channels"]
    file_format = config["dataset"]["images"]["file_format"]
    split_field = config["dataset"]["partition"]["split_field"]
    mean_list = config['prepare']['normalization']['mean']
    std_list = config['prepare']['normalization']['std']
    df = pd.read_csv(indexfile)

    tile_sizes = [256, 512, 1024]
    tile_overlaps = [0., 0.5, 0.8]

    # make directories
    os.makedirs(os.path.join(save_folder, role), exist_ok=True)
    os.makedirs(os.path.join(grid_folder, role), exist_ok=True)

    # select subgroup of dataset
    assert role.lower() in ['train', 'val', 'test', 'profile'], \
        "[Error] please choose role from ['train', 'val', 'test', 'profile']"
    if role.lower() == 'train':
        values = config["dataset"]["partition"]["training_values"]
    elif role.lower() == 'val':
        values = config["dataset"]["partition"]["validation_values"]
    else:
        values = config["dataset"]["partition"]["test_values"]

    df = df[df[split_field].isin(values)]
    df.reset_index(drop=True, inplace=True)
    print("Creating '{}' dataset ...".format(role))

    img_transforms = ['original', 'v_flip', 'h_flip', 'vh_flip', 'rot+90', 'rot-90']
    total_tile_count_list = [0, 0, 0]
    info_filename = []
    info_256 = [[], [], [], []]
    info_512 = [[], [], [], []]
    info_1024 = [[], [], [], []]

    for index, row in df.iterrows():
        print('|', end='')
        for j, img_trans in enumerate(img_transforms):
            print('.', end='')

            img_list = []
            for i_c, c in enumerate(channels):
                im = np.array(Image.open(row[c]))
                im = img_transform(im, img_trans)
                im = im / 255.
                im = (im - mean_list[i_c]) / std_list[i_c]
                img_list.append(im)
            temp_slide = np.asarray(img_list)
            temp_slide = np.moveaxis(temp_slide, 0, -1)  # shape: (xpix, ypix, channel)

            xpix, ypix, _ = temp_slide.shape

            # make folders
            temp_basefolder = os.path.join(save_folder, role, row[slide_basename]+'++'+img_trans)
            for temp_area in ['area_1', 'area_2', 'area_3', 'area_4']:
                os.makedirs(os.path.join(temp_basefolder, '1024', temp_area), exist_ok=True)
                os.makedirs(os.path.join(temp_basefolder, '512', temp_area), exist_ok=True)
                os.makedirs(os.path.join(temp_basefolder, '256', temp_area), exist_ok=True)
            os.makedirs(os.path.join(grid_folder, role, row[slide_basename] + '++' + img_trans), exist_ok=True)

            info_filename.append(row[slide_basename]+'++'+img_trans)

            for i in range(len(tile_sizes)):
                tile_size = tile_sizes[i]
                overlap = tile_overlaps[i]

                # generate grid
                temp_coords = generate_grid_4area(xpix, ypix, tile_size, overlap)
                content = {'temp_coords': temp_coords,
                           'dimension': [xpix, ypix],
                           'tile_size': tile_size,
                           'overlap': overlap}
                torch.save(content, os.path.join(grid_folder, role, row[slide_basename]+'++'+img_trans, '{}.pth'.format(tile_size)))

                # modify area id according to rotation/flip
                # 'original', 'v_flip', 'h_flip', 'vh_flip', 'rot+90', 'rot-90'
                new_coords = []
                if img_trans == 'original':
                    new_coords = temp_coords
                elif img_trans == 'v_flip':
                    new_coords = [temp_coords[1], temp_coords[0], temp_coords[3], temp_coords[2]]
                elif img_trans == 'h_flip':
                    new_coords = [temp_coords[2], temp_coords[3], temp_coords[0], temp_coords[1]]
                elif img_trans == 'vh_flip':
                    new_coords = [temp_coords[3], temp_coords[2], temp_coords[1], temp_coords[0]]
                elif img_trans == 'rot+90':
                    new_coords = [temp_coords[2], temp_coords[0], temp_coords[3], temp_coords[1]]
                elif img_trans == 'rot-90':
                    new_coords = [temp_coords[1], temp_coords[3], temp_coords[0], temp_coords[2]]
                else:
                    print('Error: area transform not recognized')

                # count tile number and save to csv
                if tile_size == 256:
                    info_256[0].append(len(new_coords[0]))
                    info_256[1].append(len(new_coords[1]))
                    info_256[2].append(len(new_coords[2]))
                    info_256[3].append(len(new_coords[3]))
                elif tile_size == 512:
                    info_512[0].append(len(new_coords[0]))
                    info_512[1].append(len(new_coords[1]))
                    info_512[2].append(len(new_coords[2]))
                    info_512[3].append(len(new_coords[3]))
                else:
                    info_1024[0].append(len(new_coords[0]))
                    info_1024[1].append(len(new_coords[1]))
                    info_1024[2].append(len(new_coords[2]))
                    info_1024[3].append(len(new_coords[3]))


                # save images
                for area_id in range(4):
                    tile_id = 0
                    for coord in new_coords[area_id]:
                        img = temp_slide[coord[0]:coord[1], coord[2]:coord[3], :].copy()
                        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
                        _ = write_tiff_file(
                            os.path.join(save_folder, role, row[slide_basename]+'++'+img_trans,
                                         str(tile_size), 'area_'+str(area_id+1), 'tile_{}.tif'.format(tile_id)), img)
                        tile_id += 1
                        total_tile_count_list[i] += 1

    # save stats to csv
    all_info = {'basename': info_filename,
                '256_area1': info_256[0],
                '256_area2': info_256[1],
                '256_area3': info_256[2],
                '256_area4': info_256[3],
                '512_area1': info_512[0],
                '512_area2': info_512[1],
                '512_area3': info_512[2],
                '512_area4': info_512[3],
                '1024_area1': info_1024[0],
                '1024_area2': info_1024[1],
                '1024_area3': info_1024[2],
                '1024_area4': info_1024[3],
                }
    df_stats = pd.DataFrame(all_info)
    df_stats.to_csv(os.path.join(grid_folder, '{}_stats.csv'.format(role)), index=False)

    print('Total tiles: {}'.format(total_tile_count_list), end='\n\n')


def make_dataset_test(indexfile, config, role, save_folder, grid_folder):
    # make the whole dataset (without data augmentation, except z-score normalization)
    # !!! update 2021 1016: new bag policy (each image will be split into 4 areas)

    slide_basename = "basename"
    channels = config["dataset"]["images"]["channels"]
    file_format = config["dataset"]["images"]["file_format"]
    split_field = config["dataset"]["partition"]["split_field"]
    mean_list = config['prepare']['normalization']['mean']
    std_list = config['prepare']['normalization']['std']
    df = pd.read_csv(indexfile)

    tile_sizes = [256, 512, 1024]
    tile_overlaps = [0., 0.5, 0.8]

    # make directories
    os.makedirs(os.path.join(save_folder, role), exist_ok=True)
    os.makedirs(os.path.join(grid_folder, role), exist_ok=True)

    # select subgroup of dataset
    assert role.lower() in ['train', 'val', 'test', 'profile'], \
        "[Error] please choose role from ['train', 'val', 'test', 'profile']"
    if role.lower() == 'train':
        values = config["dataset"]["partition"]["training_values"]
    elif role.lower() == 'val':
        values = config["dataset"]["partition"]["validation_values"]
    else:
        values = config["dataset"]["partition"]["test_values"]

    df = df[df[split_field].isin(values)]
    df.reset_index(drop=True, inplace=True)
    print("Creating '{}' dataset ...".format(role))

    img_transforms = ['original', 'v_flip', 'h_flip', 'vh_flip', 'rot+90', 'rot-90']
    total_tile_count_list = [0, 0, 0]

    for index, row in df.iterrows():
        print('|', end='')
        for j, img_trans in enumerate(img_transforms):
            print('.', end='')

            img_list = []
            for i_c, c in enumerate(channels):
                im = np.array(Image.open(row[c]))
                im = img_transform(im, img_trans)
                im = im / 255.
                im = (im - mean_list[i_c]) / std_list[i_c]
                img_list.append(im)
            temp_slide = np.asarray(img_list)
            temp_slide = np.moveaxis(temp_slide, 0, -1)  # shape: (xpix, ypix, channel)

            xpix, ypix, _ = temp_slide.shape

            # make folders
            temp_basefolder = os.path.join(save_folder, role, row[slide_basename]+'++'+img_trans)

            os.makedirs(os.path.join(temp_basefolder, '1024', 'whole'), exist_ok=True)
            os.makedirs(os.path.join(temp_basefolder, '512', 'whole'), exist_ok=True)
            os.makedirs(os.path.join(temp_basefolder, '256', 'whole'), exist_ok=True)

            os.makedirs(os.path.join(grid_folder, role, row[slide_basename] + '++' + img_trans), exist_ok=True)

            for i in range(len(tile_sizes)):
                tile_size = tile_sizes[i]
                overlap = tile_overlaps[i]
                tile_id = 0

                # grid
                temp_coords = generate_grid(xpix, ypix, tile_size, overlap)




                content = {'temp_coords': [temp_coords],
                           'dimension': [xpix, ypix],
                           'tile_size': tile_size,
                           'overlap': overlap}
                torch.save(content, os.path.join(grid_folder, role, row[slide_basename]+'++'+img_trans, '{}.pth'.format(tile_size)))

                for coord in temp_coords:
                    img = temp_slide[coord[0]:coord[1], coord[2]:coord[3], :].copy()
                    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)   # new 20211016 change to nearest neighbor
                    _ = write_tiff_file(
                        os.path.join(save_folder, role, row[slide_basename]+'++'+img_trans, str(tile_size), 'whole', 'tile_{}.tif'.format(tile_id)), img)
                    tile_id += 1
                    total_tile_count_list[i] += 1
    print('Total tiles: {}'.format(total_tile_count_list), end='\n\n')


def coords_reorder(temp_coords, img_trans):
    new_coords = []
    if img_trans == 'original':
        new_coords = temp_coords
    elif img_trans == 'v_flip':
        new_coords = [temp_coords[1], temp_coords[0], temp_coords[3], temp_coords[2]]
    elif img_trans == 'h_flip':
        new_coords = [temp_coords[2], temp_coords[3], temp_coords[0], temp_coords[1]]
    elif img_trans == 'vh_flip':
        new_coords = [temp_coords[3], temp_coords[2], temp_coords[1], temp_coords[0]]
    elif img_trans == 'rot+90':
        new_coords = [temp_coords[2], temp_coords[0], temp_coords[3], temp_coords[1]]
    elif img_trans == 'rot-90':
        new_coords = [temp_coords[1], temp_coords[3], temp_coords[0], temp_coords[2]]
    else:
        print('Error: area transform not recognized')
    return new_coords


class MILdataset(data.Dataset):
    def __init__(self, indexfile, config, role, transform=None, only_clean_sample=False, only_original=False):
        """MILdataset overwrite torch.Dataset

        !!! new bagging policy
        test is different: use whole image, choose bagging policy
        It's fine to keep merge flag to be true (in test set, every bag has only one whole SLAM slide)

        :param indexfile: (str) the path to index.csv
        :param config: (dict) configurations read from config.csv
        :param role: (str) the role of the dataset: "train" or "val" or "test"
        :param transform: (function) image transformation
        """

        # read config
        image_folder = config["dataset"]["images"]["image_folder"]
        grid_folder = config["dataset"]["images"]["grid_folder"]
        slide_basename = config["dataset"]["images"]["image_names"]
        slide_area_id = config["dataset"]["images"]["area_names"]
        channels = config["dataset"]["images"]["channels"]
        # file_format = config["dataset"]["images"]["file_format"]
        tile_size = config["dataset"]["tile"]["size"]
        split_field = config["dataset"]["partition"]["split_field"]
        merge_field = config["dataset"]["merge_object"]["merge_field"]
        target_name = config["train"]["target"]["label_field"]
        topk = config['train']['model']['top_k']

        # load index.csv
        df = pd.read_csv(indexfile)
        # new bagging policy
        df = df.sort_values(by=[merge_field])

        # identify dataset type (train/va/test)
        assert role.lower() in ['train', 'val', 'test'], \
            "[Error] please choose role from ['train', 'val', 'test']"
        if role.lower() == 'train':
            values = config["dataset"]["partition"]["training_values"]
        elif role.lower() == 'val':
            values = config["dataset"]["partition"]["validation_values"]
        else:
            values = config["dataset"]["partition"]["test_values"]

        df = df[df[split_field].isin(values)]
        if only_clean_sample:   # during test, use only images that are certain
            df = df[df['label'].isin([0, 2])]
        if only_original:   # during test, use only images that are certain
            df = df[df['transform'].isin(['original'])]
        df.reset_index(drop=True, inplace=True)
        print("Loading '{}' dataset ...".format(role))

        # slidenames = []   # store image paths, [[ch1, ch2, ...], [ch1, ch2, ...], ...]
        # slides = []   # a list of image arrays, [[im_ch1, im_ch2, ...], [im_ch1, im_ch2, ...], ...]
        grid = []   # flatten the list, a list of coordinates of tiles, [[x1, y1], [x2, y2], ...]
        slideIDX = []   # indicate the slide id for each tile [0, 0, ..., 1, 1, 1, ...]
        tile_target = []  # indicate the target for each tile, has same shape as slideIDX
        slide_target = []
        basenames = []   # the slide names [basename1, basename2, ...]
        area_ids = []
        # basenames_trans = []   # length = basenames = num_slides * num_trans   content: basename + trans
        # all_img_trans = []   # length = basenames = num_slides * num_trans
        all_tile_ids = []   # [0, 1, 2, 3, 4, 0, 1, 2, ...] length = total num of tiles
        object_ids = []  # if merge, save the object (patient/well) name here [0,0, ...,1,1,1,...] same size as slideIDX
        object_targets = []  # save targets for each object
        object_basenames = []   # dict showing object id-name pair {id:name, ...}
        object_name_dict = {}  # {name:id, ...}
        tile_object_basenames = []

        # 2022 0115
        stats_num_tile_pos = []
        stats_num_tile_neg = []

        area_dict = {'area_1': 0, 'area_2': 1, 'area_3': 2, 'area_4': 3, 'whole': 0}

        all_obj_names = list(set(df[merge_field]))
        for i, obj in enumerate(all_obj_names):
            # check if all slides from that patient have same label
            targets_obj = list(set(df.loc[df[merge_field] == obj, target_name]))
            assert len(targets_obj) == 1, "[Error] Bag '{}' has multiple targets: {}".format(obj, targets_obj)

            # add content
            obj_new_name = 'bag_' + str(obj)  # use bag id
            object_name_dict.update({obj_new_name: i})
            object_targets.append(targets_obj[0])
            object_basenames.append(obj_new_name)

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            basenames.append(row[slide_basename])
            area_id = row[slide_area_id]
            area_ids.append(area_id)

            # grid/slideIDX
            temp_grid_file = torch.load(
                os.path.join(grid_folder, role.lower(), row[slide_basename], '{}.pth'.format(tile_size)))

            temp_coords = temp_grid_file['temp_coords'][area_dict[area_id]]

            grid.extend(temp_coords)  # [[x1, x2, y1, y2], ... ]
            slideIDX.extend([index] * len(temp_coords))
            all_tile_ids.extend(list(range(len(temp_coords))))
            tile_target.extend([row[target_name]] * len(temp_coords))
            slide_target.append(row[target_name])

            obj_name = 'bag_' + str(row[merge_field])
            obj_id = object_name_dict[obj_name]
            object_ids.extend([obj_id] * len(temp_coords))
            tile_object_basenames.extend([obj_name[4:]] * len(temp_coords))

        # 2022 0115 stats number of instance
        count_pos = 0
        count_neg = 0
        for temp_i, temp_t in enumerate(object_targets):
            if temp_t == 1:
                temp_pos_num = object_ids.count(temp_i)
                stats_num_tile_pos.append(temp_pos_num)
                if temp_pos_num < topk:
                    count_pos += temp_pos_num
                else:
                    count_pos += topk
            else:
                temp_neg_count = object_ids.count(temp_i)
                stats_num_tile_neg.append(temp_neg_count)
                if temp_neg_count < topk:
                    count_neg += temp_neg_count
                else:
                    count_neg += topk

        print('|\tTotal tiles: {}\tPositive tile rate: {}'.format(len(grid), sum(tile_target)/len(tile_target)))
        print('|\tTotal bags: {}\t Positive bag rate: {}'.format(len(all_obj_names), sum(object_targets)/len(object_targets)), end='\n\n')
        print('# tile stats: \npositive bags {}|{}\nnegative bags {}|{}'.format(min(stats_num_tile_pos),
                                                                         max(stats_num_tile_pos),
                                                                         min(stats_num_tile_neg),
                                                                         max(stats_num_tile_neg)
                                                                         ))
        print('Positive rate: {}\tP({})/N({})'.format(count_pos / (count_pos + count_neg), count_pos, count_neg),
              end='\n\n')

        self.save_folder = os.path.join(image_folder, role.lower())
        self.channels = channels
        self.size = int(tile_size)
        self.basenames = basenames   # [slide1_basename, slide2_basename, ...] len=number of slides
        self.area_ids = area_ids  # same as above
        self.tile_ids = all_tile_ids
        self.targets = slide_target    # len=number of slides
        self.grid = grid  # !
        self.slideIDX = slideIDX  # indicate the slide id for each tile [0, 0, ..., 1, 1, 1, ...]
        self.tile_target = tile_target
        self.objectID = object_ids
        self.object_targets = object_targets
        self.object_basenames = object_basenames
        self.tile_object_basenames = tile_object_basenames   # only for "features_for_tsne.py"
        self.mode = None
        self.transform = transform
        self.positive_rate = count_pos / (count_pos + count_neg)

    def setmode(self, mode):
        # only affect the __getitem__ function
        # mode 1: read tiles from whole dset, FOR inference
        # mode 2: only read from t_data tiles, return target, FOR training
        # mode 3: read tiles from whole dset, return tile target, FOR validation
        self.mode = mode

    '''def maketraindata(self, idxs):
        # t_data is defined here
        self.t_data = [(self.slideIDX[x], self.tile_ids[x], self.targets[self.slideIDX[x]]) for x in idxs]'''

    def maketraindata(self, idxs_p, idxs_n):
        # t_data is defined here
        self.t_data = [(self.slideIDX[x], self.tile_ids[x], self.targets[self.slideIDX[x]]) for x in idxs_p]
        self.t_data.extend([(self.slideIDX[x], self.tile_ids[x], 0) for x in idxs_n]) # 2022 0128 bottom k

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            basename = self.basenames[slideIDX]
            area_id = self.area_ids[slideIDX]
            tile_id = self.tile_ids[index]
            img = read_tiff_file(os.path.join(self.save_folder, basename, str(self.size), area_id, 'tile_{}.tif'.format(tile_id)))
            if self.transform is not None:
                img = self.transform(img)
            return img

        elif self.mode == 2:
            slideIDX, tile_id, target = self.t_data[index]
            basename = self.basenames[slideIDX]
            area_id = self.area_ids[slideIDX]
            img = read_tiff_file(os.path.join(self.save_folder, basename, str(self.size), area_id, 'tile_{}.tif'.format(tile_id)))
            if self.transform is not None:
                img = self.transform(img)
            return img, target

        elif self.mode == 3:  # validation inference
            slideIDX = self.slideIDX[index]
            basename = self.basenames[slideIDX]
            area_id = self.area_ids[slideIDX]
            tile_id = self.tile_ids[index]
            img = read_tiff_file(os.path.join(self.save_folder, basename, str(self.size), area_id, 'tile_{}.tif'.format(tile_id)))
            target = self.tile_target[index]
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
        elif self.mode == 3:
            return len(self.grid)


def read_region(slides, coord):
    """Read a square-shaped region from slides
    :param slides: (list) a list of np arrays, each one is one channel
    :param coord: (list) the coordinates of the region, [x, y]
    :param size: (int) the size of the region
    :return tile: (ndarray) the image array for this region, with shape (size, size, channel_count)
    """
    content = []
    for slide_array in slides:
        tile_array = slide_array[coord[0]:coord[1], coord[2]:coord[3]].copy()
        '''if tile_array.shape != (size, size):
            print('[Error] grid exceed border')
            print(slidename, end='\t')
            print(slide_array.shape)
            print(coord)'''
        # tile_array = cv2.resize(tile_array, (256, 256), interpolation=cv2.INTER_LINEAR)
        tile_array = tile_array.reshape((256, 256, 1))
        content.append(tile_array)
    tile = np.concatenate(content, axis=2)  # multi-channel image, with shape: [size, size, channel_count]
    return tile


if __name__ == '__main__':
    mode = 0  # 0: generate images

    if mode == 0:
        main_folderpath = os.path.abspath('../')
        sys.path.append(main_folderpath)
        project_id = 'project0_dev'

        print('Project name: ', project_id, '\n')
        project_path = os.path.join(main_folderpath, 'projects', project_id)
        config_path = os.path.join(project_path, 'input/config/config.json')
        with open(config_path) as f:
            config = json.load(f)
            f.close()

        save_folder = '/home/derek/All_Data/2020_WSL_SLAM/All_SLAM_tiles/All_images'
        grid_folder = '/home/derek/All_Data/2020_WSL_SLAM/All_SLAM_tiles/grid_files'
        meta_path = os.path.join(os.path.abspath(project_path), config['dataset']['metadata']['meta_path'])

        # make_dataset_train_val(meta_path, config, 'train', save_folder, grid_folder)
        # make_dataset_train_val(meta_path, config, 'val', save_folder, grid_folder)
        make_dataset_test(meta_path, config, 'test', save_folder, grid_folder)

    elif mode == 1:
        # read validation datasets or demo dataset to check the the labels are right.
        a = 1






