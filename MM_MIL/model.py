"""
MIL model collection

Copyright (c) 2020 University of Illinois at Urbana-Champaign
(see LICENSE for details)
Written by Jindou Shi
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
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

# from . import dataset as mil_data


class ResNetSSL(nn.Module):

    def __init__(self, model, num_classes, in_channels=4):
        super(ResNetSSL, self).__init__()

        # set the model
        if model == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = torch.nn.Sequential()
            self.model = model
            self.fc = nn.Sequential(nn.Linear(512 * 2, 512), nn.ReLU(True), nn.Linear(512, 256))
            self.classifier = nn.Sequential(nn.Linear(256 * 3, num_classes))

        elif model == 'resnet34':
            model = models.resnet34(pretrained=False)
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = torch.nn.Sequential()
            self.model = model
            self.fc = nn.Sequential(nn.Linear(512 * 2, 512), nn.ReLU(True), nn.Linear(512, 256))
            self.classifier = nn.Sequential(nn.Linear(256*3, num_classes))

        else:
            raise NotImplementedError('not supported model type: {}'.format(model))

        self.mode = 0

    def set_mode(self, mode):
        self.mode = mode   # default mode:0 training, other mode: profiling

    def forward(self, x):
        if self.mode == 0:
            E1 = self.model(x)
            E2 = self.model(x)
            E3 = self.model(x)

            # Pairwise concatenation of features
            E12 = torch.cat((E1, E2), dim=1)
            E23 = torch.cat((E2, E3), dim=1)
            E13 = torch.cat((E1, E3), dim=1)

            E12 = self.fc(E12)
            E23 = self.fc(E23)
            E13 = self.fc(E13)

            y = torch.cat((E12, E23, E13), dim=1)
            y = self.classifier(y)
            return y
        else:
            E1 = self.model(x)
            E2 = self.model(x)
            E3 = self.model(x)

            # Pairwise concatenation of features
            E12 = torch.cat((E1, E2), dim=1)
            E23 = torch.cat((E2, E3), dim=1)
            E13 = torch.cat((E1, E3), dim=1)

            E12 = self.fc(E12)
            E23 = self.fc(E23)
            E13 = self.fc(E13)

            feature = torch.cat((E12, E23, E13), dim=1)
            y = self.classifier(feature)
            return y, feature


class ResNetSSL_Lite(nn.Module):
    """
    Only simple ResNet extractor with one classification layer
    """
    def __init__(self, model, num_classes, in_channels=4):
        super(ResNetSSL_Lite, self).__init__()

        # set the model
        if model == 'resnet18':
            model = models.resnet18(pretrained=False)
        elif model == 'resnet34':
            model = models.resnet34(pretrained=False)
        else:
            raise NotImplementedError('not supported model type: {}'.format(model))
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feature_dim = model.fc.in_features
        model.fc = torch.nn.Sequential()

        self.model = model
        self.classifier = nn.Sequential(nn.Linear(feature_dim, num_classes))
        self.mode = 0

    def set_mode(self, mode):
        self.mode = mode   # default mode:0 training, other mode: profiling

    def forward(self, x):
        if self.mode == 0:
            y = self.model(x)
            y = self.classifier(y)
            return y
        else:
            feature = self.model(x)
            y = self.classifier(feature)
            return y, feature


def resnet34(in_channels=4, class_num=2, pretrained=True, fix_weights=False):
    """ ResNet34
    :param in_channels: (int) input channel count
    :param class_num: (int) number of classes
    :param pretrained: (bool) use pretrained weights or not
    :param fix_weights: (bool) for pretrained model, fix the weights of middle layers or not
    :return: a ResNet model
    """
    model = models.resnet34(pretrained=pretrained)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, class_num)
    if pretrained and fix_weights:
        trainable_params = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'fc.weight', 'fc.bias']
        for name, param in model.named_parameters():
            if name not in trainable_params:
                param.requires_grad = False
    print('Using Pretrained ResNet34!!!')
    return model


def select_model(model, num_classes, in_channels=4, state_dict=None, lite_flag=False, fix_weight=False):

    if lite_flag:
        # only read weight for ResNet before final fc (the last 4 need to be ignored)
        model = ResNetSSL_Lite(model, num_classes, in_channels)
        model_dict = model.state_dict()

        if state_dict is not None:
            new_state_dict = OrderedDict()
            for k, v in state_dict['model'].items():
                name = k[7:]  # remove `module.`
                if name.startswith('model'):
                    new_state_dict[name] = v
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)

        if fix_weight:
            print('Model weights are fixed.')
            for name, param in enumerate(model.named_parameters()):
                if name < 108:   # only train classifier
                    param = param[1]
                    param.requires_grad = False
                else:
                    print('---- ', param[0], ' is not frozen')
    else:
        model = ResNetSSL(model, num_classes, in_channels)
        model_dict = model.state_dict()

        if state_dict is not None:
            new_state_dict = OrderedDict()
            for k, v in state_dict['model'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)

        if fix_weight:
            print('Model weights are fixed.')
            for name, param in enumerate(model.named_parameters()):
                if name < 112:  # only train classifier
                    param = param[1]
                    param.requires_grad = False
                else:
                    print('---- ', param[0], ' is not frozen')
    return model


if __name__ == '__main__':
    model = resnet34()
    for name, param in enumerate(model.named_parameters()):
        print(name, '----', param[0])
