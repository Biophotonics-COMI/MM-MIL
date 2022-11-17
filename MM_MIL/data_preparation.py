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

                # modify area id according to rotation/flip
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

                # save grid file
                content = {'temp_coords': new_coords,
                           'dimension': [xpix, ypix],
                           'tile_size': tile_size,
                           'overlap': overlap}
                torch.save(content, os.path.join(grid_folder, role, row[slide_basename] + '++' + img_trans,
                                                 '{}.pth'.format(tile_size)))

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
    # tile_overlaps = [0., 0.5, 0.8]
    tile_overlaps = [0.9, 0.9, 0.9]

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


# 2022 0305 coverage ratio
def make_dataset_test_certain_cancer(indexfile, config, role, save_folder, grid_folder):
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
    # tile_overlaps = [0., 0.5, 0.8]
    tile_overlaps = [0.9, 0.9, 0.9]

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
    df = df[df['label_whole'].isin([2])]   # only certain cancer
    print(df.shape)

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


# 2022 0305 coverage ratio
def make_dataset_test_certain_cancer_specific(indexfile, config, role, save_folder, grid_folder):
    # make the whole dataset (without data augmentation, except z-score normalization)
    # !!! update 2021 1016: new bag policy (each image will be split into 4 areas)

    slide_basename = "basename"
    channels = config["dataset"]["images"]["channels"]
    file_format = config["dataset"]["images"]["file_format"]
    split_field = config["dataset"]["partition"]["split_field"]
    mean_list = config['prepare']['normalization']['mean']
    std_list = config['prepare']['normalization']['std']
    df = pd.read_csv(indexfile)

    b_names_all = []
    b_img_transforms = ['original', 'v_flip', 'h_flip', 'vh_flip', 'rot+90', 'rot-90']
    b_names_only = ['20170825_human breast cancer_1p8mm_FOV450um_s1_m1_z170_10us',
                    '20170911_human normal_1p2mm_FOV500um_s2_m2_z170_10us'
                    ]
    for bt in b_img_transforms:
        for bn in b_names_only:
            b_names_all.append(bn+'++'+bt)

    tile_sizes = [256, 512, 1024]
    # tile_overlaps = [0., 0.5, 0.8]
    tile_overlaps = [0.9, 0.9, 0.9]

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

    df = df[df['basename'].isin(b_names_only)]
    print(df.shape)

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


def fix_grid_train_val(indexfile, config, role, save_folder, grid_folder):
    # make the whole dataset (without data augmentation, except z-score normalization)
    # !!! update 2021 1016: new bag policy (each image will be split into 4 areas)

    slide_basename = 'basename'
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
    info_filename = []

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
            temp_basefolder = os.path.join(save_folder, role, row[slide_basename] + '++' + img_trans)
            for temp_area in ['area_1', 'area_2', 'area_3', 'area_4']:
                os.makedirs(os.path.join(temp_basefolder, '1024', temp_area), exist_ok=True)
                os.makedirs(os.path.join(temp_basefolder, '512', temp_area), exist_ok=True)
                os.makedirs(os.path.join(temp_basefolder, '256', temp_area), exist_ok=True)
            os.makedirs(os.path.join(grid_folder, role, row[slide_basename] + '++' + img_trans), exist_ok=True)

            info_filename.append(row[slide_basename] + '++' + img_trans)

            for i in range(len(tile_sizes)):
                tile_size = tile_sizes[i]
                overlap = tile_overlaps[i]

                # generate grid
                temp_coords = generate_grid_4area(xpix, ypix, tile_size, overlap)

                # modify area id according to rotation/flip
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

                # save grid file
                content = {'temp_coords': new_coords,
                           'dimension': [xpix, ypix],
                           'tile_size': tile_size,
                           'overlap': overlap}
                torch.save(content, os.path.join(grid_folder, role, row[slide_basename] + '++' + img_trans,
                                                 '{}.pth'.format(tile_size)))


if __name__ == '__main__':
    mode = 0  # 0: generate images

    if mode == 0:
        main_folderpath = os.path.abspath('../')
        sys.path.append(main_folderpath)
        project_id = 'CrossVal1_256'

        print('Project name: ', project_id, '\n')
        project_path = os.path.join(main_folderpath, 'projects', project_id)
        config_path = os.path.join(project_path, 'input/config/config.json')
        with open(config_path) as f:
            config = json.load(f)
            f.close()

        # 2022 0428 cross validation (5-fold Monte Carlo)
        save_folder = '/home/derek/NewStorage/2022_WSL_SLAM/All_SLAM_tiles/All_images'
        grid_folder = '/home/derek/NewStorage/2022_WSL_SLAM/All_SLAM_tiles/grid_files'
        meta_path = os.path.join(os.path.abspath(project_path), 'input/metadata/all_new_index.csv')

        # make_dataset_train_val(meta_path, config, 'train', save_folder, grid_folder)
        # make_dataset_train_val(meta_path, config, 'val', save_folder, grid_folder)
        make_dataset_test_certain_cancer_specific(meta_path, config, 'test', save_folder, grid_folder)
        # fix_grid_train_val(meta_path, config, 'train', save_folder, grid_folder)
        # fix_grid_train_val(meta_path, config, 'val', save_folder, grid_folder)

    elif mode == 1:
        # read validation datasets or demo dataset to check the the labels are right.
        a = 1


