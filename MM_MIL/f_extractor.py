"""
Use trained model as feature extractor
Written Jindou Shi
"""

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

import MM_MIL.model as mil_model
import MM_MIL.dataset as mil_data


def f_extract(config, project_path, save_dir, train_result=None):
    """
    :param config: (dict) configurations read from config.json
    :param project_path: (str) folder path of this project, which contain 'input' and 'output' folder
    :param save_dir: (str) the folder to save all the features
    :param train_result: (str) the folder name of training output, by default using the latest output
    :return:
    """

    meta_path = os.path.join(os.path.abspath(project_path), config['dataset']['metadata']['meta_path'])
    merge_flag = config["dataset"]["merge_object"]["flag"]
    model_name = config['train']['model']['name']
    nepochs = config['train']['model']['epochs']
    batch_size = config['train']['model']['params']['batch_size']
    lr = config['train']['model']['params']["learning_rate"]
    weight_decay = config['train']['model']['params']["weight_decay"]
    k = config['train']['model']['top_k']
    pos_weights = config['train']['criterion']['positive_class_weights']
    workers = config['train']['load_worker']

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
    model = mil_model.resnet34()

    if torch.cuda.device_count() > 1:
        state_dict = fix_state_dict(ch['state_dict'])
    else:
        state_dict = ch['state_dict']

    model.load_state_dict(state_dict)
    model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.cuda()
    cudnn.benchmark = True
    print('Loading weights from training epoch {}'.format(ch['epoch']))

    '''normalization'''
    mean_list, std_list = ch['normalization']
    normalize = transforms.Normalize(mean=mean_list, std=std_list)
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    '''load dataset'''
    config_modified = config
    # config_modified['dataset']['partition']['test_values'] = [1, 2, 3]
    config_modified['dataset']['merge_object']['merge_field'] = "b++trans"
    dset = mil_data.MILdataset(meta_path, config_modified, 'test', None, False, only_original=False)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False)
    dset.setmode(1)

    '''model inference'''
    probs, features = inference(config, loader, model)  # feature: 512 features for each tile

    '''save tile probabilities/features'''
    new_list = tile_probabilities(dset, probs)

    # save features in separate files
    # save_features(features, new_list, dset, save_dir)

    # save all features in one file:
    temp_dict = {'features': features, 'probs_all': probs, 'prob_separate': new_list}
    torch.save(temp_dict, os.path.join(save_dir, 'all_features.pt'))
    print('\n=========== Done! ===========\n\n')


def save_features(features, probs, dset, save_dir):
    """
    Save feature embeddings for each tile
    :param features: (ndarray) the feature tensor for all tiles (number of tiles, number of features)
    :param probs: (list) the predicted probabilities of tiles
    :param dset: (MILdataset) the MIL dataset
    :param save_dir: (str) the folder to save all the feature embedding files
    :return:
    """
    slideIDX = dset.slideIDX   # indicate the slide id for each tile [0, 0, ..., 1, 1, 1, ...]
    b_names = dset.basenames   # [slide1_basename, slide2_basename, ...] len=number of slides

    for i in range(max(slideIDX)+1):
        b_name = b_names[i]
        start_index = slideIDX.index(i)
        end_index = len(slideIDX) - 1 - slideIDX[::-1].index(i)

        temp_feature_array = features[start_index:end_index+1, :].copy()
        temp_probs = probs[i].copy()
        assert len(temp_probs) == end_index - start_index + 1, 'feature array does not match with probs'

        temp_dict = {'features': temp_feature_array, 'probs': temp_probs}
        torch.save(temp_dict, os.path.join(save_dir, b_name+'_feature.pt'))


def inference(config, loader, model):
    """
    Inference the model using loader dataset, generate probability for each tile
    :param config: (dict) configuration info read from config.json
    :param loader: (torch.DataLoader) each element is one batch
    :param model: (torchvision.models) model
    :return probs: (numpy array) the probability of each tile
    """
    f_length = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048}
    batch_size = config['train']['model']['params']['batch_size']
    model_name = config['train']['model']['name']
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    features = torch.FloatTensor(len(loader.dataset), f_length[model_name])
    print('Testing model\tBatches:')
    with torch.no_grad():
        for i, input in enumerate(tqdm(loader, ncols=100)):
            input = input.float()
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*batch_size:i*batch_size+input.size(0)] = output.detach()[:, 1].clone()
            # extract activation values in middle layer
            features[i*batch_size:i*batch_size+input.size(0), :] = activation['avgpool'].detach().squeeze().clone()
    return probs.cpu().numpy(), features.cpu().numpy()


def fix_state_dict(state_dict):
    """remove .module from keys, caused by nn.DataParallel"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


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


if __name__ == '__main__':
    main_folderpath = os.path.abspath('../')

    print('************ MILclassifier--Profiling ************')
    print('Copyright (c) 2020 UIUC\nWritten by Jindou Shi')
    print('**************************************************')

    project_id = 'project1_test_topK_256'
    train_process = 'TrainingProcess_20211023-180746'
    save_dir = os.path.join(main_folderpath, 'projects', project_id, 'output', train_process, 'features')
    os.makedirs(save_dir, exist_ok=True)

    print('Project name', project_id, '\n')

    project_path = os.path.join(main_folderpath, 'projects', project_id)
    config_path = os.path.join(project_path, 'input/config/config.json')
    with open(config_path) as f:
        config = json.load(f)
        f.close()

    f_extract(config, project_path, save_dir, train_result=train_process)


