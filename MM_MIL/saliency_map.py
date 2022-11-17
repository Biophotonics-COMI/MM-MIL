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

from captum.attr import (
    GradientShap,
    Saliency,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)


def saliency_map(config, project_path, check_point='best', train_result=None,
                          only_clean_sample=False):
    """
    :param config: (dict) configurations read from config.json
    :param project_path: (str) folder path of this project, which contain 'input' and 'output' folder
    :param train_result: (str) the folder name of training output, by default using the latest output
    :return:
    """

    print('\n********** Test mode **********\n')

    meta_path = os.path.join(os.path.abspath(project_path), config['dataset']['metadata']['meta_path'])
    workers = config['train']['load_worker']
    batch_size = 1

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
    config_modified = config
    config_modified['dataset']['merge_object']['merge_field'] = "b++trans"
    dset = mil_data.MILdataset(meta_path, config_modified, 'test', None, only_clean_sample, only_original=True)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=False)
    dset.setmode(1)

    '''generate saliency map'''
    saliency_maps = np.float16(generate_saliency_map_SmoothGrad(config, batch_size, loader, model))
    torch.save({'saliency_map': saliency_maps, 'tile_slide_name': dset.tile_object_basenames, 'tile_id': dset.tile_ids},
               os.path.join(train_result_path, 'predictions', 'SaliencyMap_SmoothGrad.pt'),
               pickle_protocol=4)

    print('\n=========== Done! ===========\n\n')


def generate_saliency_map_SmoothGrad(config, batch_size, loader, model):
    # batch_size = config['train']['model']['params']['batch_size']
    # batch_size = 1
    channels = config["dataset"]["images"]["channels"]
    blank_mean = [-1.0507776252462933, -0.7785514434029203, -0.9453880760635832, -0.5086745930855703]
    blank_std = [0.032617591007461484, 0.0672731448631288, 0.08720297676771936, 0.06543926817717291]
    temp_blank_img0 = np.random.normal(loc=blank_mean[0], scale=blank_std[0], size=(1, 256, 256))
    temp_blank_img1 = np.random.normal(loc=blank_mean[1], scale=blank_std[1], size=(1, 256, 256))
    temp_blank_img2 = np.random.normal(loc=blank_mean[2], scale=blank_std[2], size=(1, 256, 256))
    temp_blank_img3 = np.random.normal(loc=blank_mean[3], scale=blank_std[3], size=(1, 256, 256))
    blank_img = np.concatenate((temp_blank_img0, temp_blank_img1, temp_blank_img2, temp_blank_img3), axis=0)
    blank_img = np.reshape(blank_img, (1, 4, 256, 256))
    blank_img = np.float16(blank_img)

    model.eval()
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)

    s_maps = torch.FloatTensor(len(loader.dataset), len(channels), 256, 256)  # number of tiles, channel,
    print('Generating saliency map\tBatches:')

    for i, input in enumerate(tqdm(loader, ncols=100)):
        input = input.float()
        # aaa = input.size()  # (1, 4, 256, 256)
        # baseline = torch.zeros(input.size())
        baseline = torch.from_numpy(blank_img)   # 2022 0208 change baseline
        baseline = baseline.cuda()
        input = input.cuda()
        input = input.requires_grad_()

        # 1. IntegratedGradients
        attributions = nt.attribute(input, nt_type='smoothgrad', stdevs=0.02, n_samples=4,
                                    baselines=baseline, target=1, return_convergence_delta=False)

        s_maps[i * batch_size:i * batch_size + input.size(0), :, :, :] = attributions.clone()
    return s_maps.cpu().numpy()


def fix_state_dict(state_dict):
    """remove .module from keys, caused by nn.DataParallel"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

