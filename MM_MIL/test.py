import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import PIL.Image as Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from . import model as mil_model
from . import utils as mil_utils
from . import dataset as mil_data
from . import evaluation as mil_eval


def model_inference(config, project_path, result_path,
                    dset_names=['train', 'val', 'test'], check_point='final', train_result=None, only_clean_sample=True,
                    nbins=10):
    """
    :param config: (dict) configurations read from config.json
    :param project_path: (str) folder path of this project, which contain 'input' and 'output' folder
    :param train_result: (str) the folder name of training output, by default using the latest output
    :return:
    """

    print('\n********** Inference mode (for bag aggregation) **********\n')

    meta_path = os.path.join(os.path.abspath(project_path), config['dataset']['metadata']['meta_path'])
    batch_size = config['test']['batch_size']
    workers = config['train']['load_worker']

    '''create output folder'''
    save_folders = ['1_input']
    for sf in save_folders:
        os.makedirs(os.path.join(result_path, train_result, sf), exist_ok=True)

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

    if check_point == 'final':
        ch = torch.load(os.path.join(train_result_path, 'weights', 'checkpoint_final.pth'))
    else:
        ch = torch.load(os.path.join(train_result_path, 'weights', 'checkpoint_best.pth'))

    '''load model'''
    model = mil_model.resnet34()

    if torch.cuda.device_count() > 1:
        state_dict = fix_state_dict(ch['state_dict'])
    else:
        state_dict = ch['state_dict']

    model.load_state_dict(state_dict)
    model.cuda()
    cudnn.benchmark = True
    print('Mode: {} \nLoading weights from training epoch {}'.format(check_point, ch['epoch']))

    '''load dataset'''
    for dset_name in dset_names:
        print('Inferencing: {} set'.format(dset_name))
        dset = mil_data.MILdataset(meta_path, config, dset_name, None, only_clean_sample)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=False)
        dset.setmode(1)

        '''model inference'''
        probs = inference(config, loader, model)  # feature: 512 features for each tile

        '''save slide-/object-level predictions'''
        maxs = group_max(np.array(dset.objectID), probs, max(dset.objectID) + 1)
        thresholdOpt, gmeanOpt, fprOpt, tprOpt = mil_eval.select_optimal_threshold(dset.object_targets, maxs)

        num_pos_tile, total_tile, perc_pos_tile, hist_features = prob_feature_extraction(dset, probs, nbins,
                                                                                         prob_thres=0.5)

        pred = [1 if x >= thresholdOpt else 0 for x in maxs]
        err, fpr, fnr = calc_err(pred, dset.object_targets)
        print(
            'Test result: Accuracy: {}%\tOptimal_thres: {}\tFPR: {}\tFNR: {}'.format((1 - err) * 100, thresholdOpt, fpr,
                                                                                     fnr))

        # create dataframe
        all_info = {'Info_filename': dset.object_basenames,
                    'Info_target': dset.object_targets,
                    'num_pos_tile': num_pos_tile,
                    'total_tile': total_tile,
                    'perc_pos_tile': perc_pos_tile,
                    'max_probability': maxs
                    }
        for b_id in range(nbins):
            all_info.update({'hist_{}'.format(b_id): list(hist_features[:, b_id])})

        df_result = pd.DataFrame(all_info)
        df_result.to_csv(os.path.join(result_path, train_result, '1_input', '{}_{}_features.csv'.format(check_point, dset_name)),
                         index=False)

        '''save tile probabilities/features'''
        # new_list = tile_probabilities(dset, probs)
        # new_list_dict = {'probs': new_list, 'optimal_thres': thresholdOpt}
        # torch.save(new_list_dict, os.path.join(train_result_path, '1_input', '{}_{}_predictions_tile_prob.pt'.format(dset_name, check_point)))
    print('\n=========== Done! ===========\n\n')


def inference(config, loader, model):
    """inference the model using loader dataset, generate probability for each tile
    :param config: (dict) configuration info read from config.json
    :param loader: (torch.DataLoader) each element is one batch
    :param model: (torchvision.models) model
    :return probs: (numpy array) the probability of each tile
    """
    f_length = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048}
    f_length_lite = {'resnet18': 768, 'resnet34': 768}   # 256 * 3

    batch_size = config['test']['batch_size']

    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))

    print('Testing model\tBatches:')
    with torch.no_grad():
        for i, input in enumerate(tqdm(loader, ncols=100)):
            input = input.float()
            input = input.cuda()
            output = model(input)
            output = F.softmax(output, dim=1)
            probs[i * batch_size:i * batch_size + input.size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def prob_feature_extraction(dset, probs, nbins=10, prob_thres=0.5):
    """
    Tile probability based feature extraction

    count how many positive tiles per slide
    :param objectIDX: (1d array)  an array of dset.slideIDX, with shape (total_tile_count,)
    :param probs: (1d array) an array of probabilities, with same shape as slideIDX
    :return:
    """

    objectIDX = dset.objectID

    num_pos_tile = []
    total_tile = []
    perc_pos_tile = []
    hist_features = []

    objectIDX = list(objectIDX)
    all_index = set(objectIDX)

    for i in all_index:
        idx_0 = objectIDX.index(i)  # find the range of tiles belong to this slide
        idx_1 = len(objectIDX) - 1 - objectIDX[::-1].index(i)
        temp_probs = probs[idx_0:idx_1]   # !!! all instance probabilities
        num_all = len(temp_probs)
        num_pos = np.sum(temp_probs > prob_thres)

        # 1. feature: percentage of positive (or prob>0.5) instances in a bag
        num_pos_tile.append(num_pos)
        total_tile.append(num_all)
        perc_pos_tile.append(float(num_pos)/num_all)

        # 2. feature: histogram (PDF function)
        hist, bin_edges = np.histogram(temp_probs, range=(0.0, 1.0), bins=nbins, density=False)
        hist_features.append(hist / np.sum(hist))
        a =1

    return num_pos_tile, total_tile, perc_pos_tile, np.array(hist_features)


def calc_err(pred, real):
    """err, fpr, fnr = calc_err(pred, val_dset.targets)
    :param pred: (list) the final prediction for each slide
    :param real: (list) ground truth, has the same length as pred
    :return: error rate, fpr, fnr
    """
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)  # *Return (x1 != x2) element-wise. An array with the same shape as pred/real
    err = float(neq.sum()) / pred.shape[0]
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum()
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
    return err, fpr, fnr


def fix_state_dict(state_dict):
    """remove .module from keys, caused by nn.DataParallel"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def tile_probabilities(dset, probs):
    """Generate a list of tile probabilities for each slide
    :param dset: (MILdataset) the MILdataset
    :param probs: (np array) the output of inference
    :return new_list: (list) a list of tile probabilities for each slide, with length=number of slides
    """
    start_idx = 0
    new_list = []  # [[tile probabilities for slide 1], [tile probabilities for slide 2], ...]
    idx_list = dset.slideIDX
    for i in range(max(idx_list) + 1):
        length = idx_list.count(i)
        temp_list = probs[start_idx:start_idx + length]
        start_idx = start_idx + length
        new_list.append(temp_list)
    return new_list


def group_max(groups, data, nmax):
    """Generate a list of probs for slides, one prob for one slide
    :param groups: (1d array)  an array of dset.slideIDX, with shape (total_tile_count,)
    :param data: (1d array) an array of probabilities, with same shape as groups
    :param nmax: (int) target number,  equals slide number
    :return:
    """
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))   # Sort first on data and then groups
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return list(out)




