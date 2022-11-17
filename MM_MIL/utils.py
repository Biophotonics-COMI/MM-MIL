"""
Common utility functions and classes.

Copyright (c) 2020 University of Illinois at Urbana-Champaign
(see LICENSE for details)
Written by Jindou Shi
"""

import os
import sys
import json
import math
import time
import numpy as np
import pandas as pd
from os import listdir
import PIL.Image as Image
from skimage import filters
import matplotlib.pyplot as plt


def otsu_mask(img):
    """ Generate Otsu mask
    :param img: (2d array) a grayscale image
    :return mask (2d array): a binary mask
    """
    thres = filters.threshold_otsu(img)
    mask = np.uint8(img < thres)
    return mask


def grid_generation(mask, tile_size, overlap=0.0, thres=0.2):
    """ Generate grid, a list of coordinates for tiles
    :param mask: (2d array) a binary mask for original slide. Can be an all-one array, with no region to ignore.
    :param tile_size: (int) the size of tiles (in pixels), the shape of tiles will be (tile_size, tile_size)
    :param overlap: (float) the percentage of tile overlap, e.g. overlap=0.5 for 50% overlap between tiles
    :param thres: (float) the threshold to remove tiles with little content, e.g. thres=0.2 removes the tiles with \
    less than 20% of the area covered by ones in the mask
    :return grid: (list) a list of coordinates of tiles
    """
    assert len(mask.shape) == 2, "[ERROR] The mask should have only two dimensions, with the same shape as the slide"
    assert (overlap < 1) and (overlap >= 0), "[ERROR] The range of overlap ratio should be [0.0, 1.0)"
    print('.', end='')
    # generate grid
    pix_x, pix_y = mask.shape
    grid_list = []
    length = int(tile_size)
    ol_len = length * (1 - overlap)
    count_x = math.floor(1 + (pix_x - length) / float(ol_len))
    count_y = math.floor(1 + (pix_y - length) / float(ol_len))
    for i in range(count_x):
        for j in range(count_y):
            temp_x = int(i * ol_len)
            temp_y = int(j * ol_len)
            assert (temp_x+length) <= pix_x and (temp_y+length) <= pix_y, "[ERROR] grid exceed border"
            temp_mask = mask[temp_x:temp_x + length, temp_y:temp_y + length].copy()
            sum_ones = np.sum(temp_mask)
            ones_ratio = sum_ones / float(length) ** 2.0
            if ones_ratio > thres:
                grid_list.append([temp_x, temp_y])
    return grid_list


def save_grid(grid_list, save_path):
    """ Save grid_list and tile_size to a json file
    :param grid_list: (list) the gird list, containing all the coordinates of tiles
    :param save_path: (str) the path where to save the file, should include filename which ends with ".json"
    """
    json_content = {
        "grids": grid_list
    }
    with open(save_path, 'w') as json_file:
        json.dump(json_content, json_file)


def find_file(folder_path, file_format='', keyword=''):
    """ Find all files in the folder and return a file list
    :param folder_path: (str) the path of the folder
    :param file_format: (str) the file type (e.g. file_format='png')
    :param keyword: (str) the key word in the targeted file title, can be upper- or lower-cased
    :return file_list: (list) a list of files found
    """

    if file_format == '':
        file_list = listdir(folder_path)
    else:
        if not file_format.startswith('.'):
            file_format = '.' + file_format
        file_list = [f for f in listdir(folder_path) if os.path.splitext(f)[-1] == file_format]
    if keyword != '':
        new_file_list = []
        keyword = keyword.lower()
        for file in file_list:
            lower_file = file.lower()
            if keyword in lower_file:
                new_file_list.append(file)
        file_list = new_file_list
    return file_list


def save_dict2csv(data, save_path, sort=None):
    """Save a dictionary to csv file
    :param data: (dict) a dictionary containing all the info: e.g.{'feature1': [1,2,3], 'feature2': [5,6,7]}
    :param save_path: (str) the save path
    :param sort: (bool) sort DF by the first column
    """
    col = list(data.keys())
    df = pd.DataFrame(data, columns=col)
    if sort is not None:
        df = df.sort_values(by=sort)
    df.to_csv(save_path, index=False)


def image_stats(img):
    """
    Calculate stats for an image (mean, std, max, min)
    :param img: (ndarray) image array with shape (xpix, ypix, num_dimension)
    :return ch_stats: (array) stats for each channel: [[stats for channel_1], [stats for channel_2], ...]
    """
    assert len(img.shape) in [2, 3], '[Error] input image has wrong shape: {}'.format(img.shape)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    # mean, std, max, min
    ch_stats = []
    for i in range(img.shape[2]):
        stats = []
        im = img[:, :, i]
        stats.append(np.mean(im))
        stats.append(np.std(im))
        stats.append(np.max(im))
        stats.append(np.min(im))
        ch_stats.append(stats.copy())
    return np.array(ch_stats)


def generate_stats_report(config, meta_path, save_flag=True, norm_flag=True):
    """Generate stats report for whole dataset
    !!! designed only for grayscale images used in our study, with channels specified in metadata
    Please customize the stats report generation accordingly

    :param config: (dict) configurations
    :param meta_path: (str) the path to index.csv
    :param save_flag: (bool) save the stats or not, if yes, save all stats to the same folder of index.csv
    :param norm_flag: (bool) whether images are normalized to 0~1 in MILdataset
    :return mean_list: (list) mean for each channel, len(mean_list) == channel count
    :return std_list: (list) std for each channel, len(std_list) == channel count
    """
    stats_names = ['mean', 'std', 'max', 'min']
    channels = config["dataset"]["images"]["channels"]

    df = pd.read_csv(meta_path)
    all_stats = [[[] for i in range(len(stats_names))] for j in range(len(channels))]   # shape: (#stats, #channels)
    for index, row in df.iterrows():
        for i, c in enumerate(channels):
            img = np.array(Image.open(row[c]))
            stats = np.squeeze(image_stats(img))   # shape: (4,)
            for j, element in enumerate(stats):
                all_stats[i][j].append(stats[j])
    all_dict = {}
    for i, c in enumerate(channels):
        c_name = '{}_'.format(c)
        for j, name in enumerate(stats_names):
            key = c_name + name
            value = all_stats[i][j]
            all_dict.update({key: value})
    col = list(all_dict.keys())
    df = pd.DataFrame(all_dict, columns=col)

    if save_flag:
        # save the stats.csv to the same folder where the index.csv is saved
        folder_path = os.path.dirname(meta_path)
        save_path = os.path.join(folder_path, 'stats.csv')
        save_dict2csv(all_dict, save_path, sort=None)

    mean_list = []
    std_list = []
    for c in channels:
        mean_key = c + '_mean'
        std_key = c + '_std'
        max_key = c + '_max'
        min_key = c + '_min'
        if norm_flag:
            # images are normed t 0~1 in MILDataset, while stats.csv not modified
            # 1. generate normalized mean and std
            mean_norm_key = mean_key + '_norm'
            std_norm_key = std_key + '_norm'
            df[mean_norm_key] = (df[mean_key] - df[min_key]) / (df[max_key] - df[min_key])
            df[std_norm_key] = df[std_key] / (df[max_key] - df[min_key])
            # 2. simply average (not weighted on image size)
            mean_list.append(np.mean(df[mean_norm_key]))
            std_list.append(np.mean(df[std_norm_key]))
        else:
            mean_list.append(np.mean(df[mean_key]))
            std_list.append(np.mean(df[std_key]))
    return mean_list, std_list


def combined_mean_std(config, stats_path, norm_flag=True):
    """determine mean, std for each channel of all images combined
    :param config: (dict) configurations
    :param stats_path: (str) the path to stats.csv file
    :param norm_flag: (bool) True: images in MILdataset have already been normalized to 0~1
                             False: images is not normalized, with max and min value equal to values in stats
    :return mean_list: (list) mean for each channel, len(mean_list) == channel count
    :return std_list: (list) std for each channel, len(std_list) == channel count
    """
    stats_names = ['mean', 'std', 'max', 'min']
    channels = config["dataset"]["images"]["channels"]
    df = pd.read_csv(stats_path)
    mean_list = []
    std_list = []
    for c in channels:
        mean_key = c + '_mean'
        std_key = c + '_std'
        max_key = c + '_max'
        min_key = c + '_min'
        if norm_flag:
            # images are normed t 0~1 in MILDataset, while stats.csv not modified
            # 1. generate normalized mean and std
            mean_norm_key = mean_key + '_norm'
            std_norm_key = std_key + '_norm'
            df[mean_norm_key] = (df[mean_key] - df[min_key]) / (df[max_key] - df[min_key])
            df[std_norm_key] = df[std_key] / (df[max_key] - df[min_key])
            # 2. simply average (not weighted on image size)
            mean_list.append(np.mean(df[mean_norm_key]))
            std_list.append(np.mean(df[std_norm_key]))
        else:
            mean_list.append(np.mean(df[mean_key]))
            std_list.append(np.mean(df[std_key]))
    return mean_list, std_list


def export_train_report(log_path):
    """ generate plot for training and validation process
    :param log_path: (str) the path to project log folder, where convergence.csv is saved
    """
    log_file_path = os.path.join(log_path, 'convergence.csv')
    df = pd.read_csv(log_file_path)
    df1 = df.dropna()

    # loss (train/val)
    fig, axs = plt.subplots(2, 1, figsize=(7, 9))
    color = 'tab:blue'
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('training loss', color=color)
    axs[0].plot(df['epoch'], df['train_loss'], color=color)
    axs[0].tick_params(axis='y', labelcolor=color)
    axs[0].set_title('Training/Validation Loss')
    ax2 = axs[0].twinx()
    color = 'tab:red'
    ax2.set_ylabel('validation loss', color=color)
    ax2.plot(df1['epoch'], df1['val_loss'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)


    # validation acc/fpr/fnr
    axs[1].plot(df1['epoch'], df1['val_acc'], 'ro-', label='Accuracy')
    axs[1].plot(df1['epoch'], df1['val_fpr'], 'b+-', label='False Positive Rate')
    axs[1].plot(df1['epoch'], df1['val_fnr'], 'g+-', label='False Negative Rate')
    axs[1].legend(loc='center',  bbox_to_anchor=(0.5, -0.2), ncol=3)
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('rate')
    axs[1].set_title('Validation Performance')

    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(log_path, 'training_results.png'))
    plt.close()



"""
def calculate_mean_std(config, project_path, train_result=None):
    
    print('\n********** Test mode **********\n')

    meta_path = os.path.join(os.path.abspath(project_path), config['dataset']['metadata']['meta_path'])
    merge_flag = config["dataset"]["merge_object"]["flag"]
    model_name = config['train']['model']['name']
    nepochs = config['train']['model']['epochs']

    config['train']['model']['params']['batch_size'] = 1
    batch_size = config['train']['model']['params']['batch_size']
    # batch_size = config['test']['batch_size']

    lr = config['train']['model']['params']["learning_rate"]
    weight_decay = config['train']['model']['params']["weight_decay"]
    k = config['train']['model']['top_k']
    pos_weights = config['train']['criterion']['positive_class_weights']
    workers = config['train']['load_worker']
    test_every = config['train']['validation']['test_every']
    add_val = config['train']['validation']['add_val']

    '''load training results'''
    output_path = os.path.join(project_path, 'output')
    all_results = [f for f in os.listdir(output_path) if 'TrainingProcess' in f]
    if train_result is None:
        train_result = max(all_results)
    else:
        assert train_result in all_results, \
            '[Error] The given train_result folder cannot be found. Please choose from {}'.format(all_results)
    print("loading: {}".format(train_result))
    train_result_path = os.path.join(output_path, train_result)
    ch = torch.load(os.path.join(train_result_path, 'weights', 'checkpoint_best.pth'))

    '''load model'''
    model = mil_model.select_model(config)
    print("Trained model: {}".format(model_name))

    if torch.cuda.device_count() > 1:
        state_dict = fix_state_dict(ch['state_dict'])
    else:
        state_dict = ch['state_dict']

    model.load_state_dict(state_dict)
    model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.cuda()
    cudnn.benchmark = True
    print('Loading weights from training epoch {}'.format(ch['epoch']))

    normalization
    mean_list, std_list = ch['normalization']
    normalize = transforms.Normalize(mean=mean_list, std=std_list)
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    '''load dataset'''
    dset = mil_data.MILdataset(meta_path, config, 'test', trans)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False)
    dset.setmode(1)

    '''model inference'''
    probs, features = inference(config, loader, model)  # feature: 512 features for each tile

"""


def calculate_mean_std(config, meta_path):
    """Generate stats report for whole dataset
    !!! designed only for grayscale images used in our study, with channels specified in metadata
    Please customize the stats report generation accordingly

    :param config: (dict) configurations
    :param meta_path: (str) the path to index.csv
    :param save_flag: (bool) save the stats or not, if yes, save all stats to the same folder of index.csv
    :param norm_flag: (bool) whether images are normalized to 0~1 in MILdataset
    :return mean_list: (list) mean for each channel, len(mean_list) == channel count
    :return std_list: (list) std for each channel, len(std_list) == channel count
    """
    stats_names = ['mean', 'std', 'max', 'min']
    channels = config["dataset"]["images"]["channels"]

    df = pd.read_csv(meta_path)

    mean = np.zeros(len(channels))
    num_sample = 0.

    for index, row in df.iterrows():
        num_sample += 1
        for i, c in enumerate(channels):
            img = np.array(Image.open(row[c])) / 255.
            mean[i] += np.mean(img)

    mean /= num_sample
    variance = np.zeros(len(channels))

    for index, row in df.iterrows():
        for i, c in enumerate(channels):
            img = np.array(Image.open(row[c])) / 255.
            h, w = img.shape
            temp = np.sum(np.square(img - mean[i])) / (h * w)
            variance[i] += temp
    std = np.sqrt(variance / num_sample)   # num of channels
    return mean, std


if __name__ == '__main__':
    main_folderpath = os.path.abspath('../')
    sys.path.append(main_folderpath)

    project_id = 'project1_tile_size_256'

    print('Project name', project_id, '\n')
    project_path = os.path.join(main_folderpath, 'projects', project_id)
    config_path = os.path.join(project_path, 'input/config/config_redox_imagej.json')
    with open(config_path) as f:
        config = json.load(f)
        f.close()
    meta_path = os.path.join(project_path, 'input/metadata/index_redox_imagej.csv')

    time_0 = time.time()
    mean_list, std_list = calculate_mean_std(config, meta_path)
    time_1 = time.time()

    print(mean_list, std_list)
    print(time_1 - time_0)























