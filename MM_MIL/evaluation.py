"""
Evaluation functions

Copyright (c) 2020 University of Illinois at Urbana-Champaign
(see LICENSE for details)
Written by Jindou Shi
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt

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

from . import model as mil_model
from . import utils as mil_utils
from . import dataset as mil_data


def validation_roc_auc_thresopt(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # 20210819: add G_mean to select best threshold for this binary classification task
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    gmeanOpt = round(gmean[index], ndigits=4)
    fprOpt = round(fpr[index], ndigits=4)
    tprOpt = round(tpr[index], ndigits=4)

    preds = [1 if x >= thresholdOpt else 0 for x in y_score]
    err, _, _ = calc_err(preds, y_true)

    return roc_auc, thresholdOpt, gmeanOpt, fprOpt, 1-tprOpt, 1-err


def evaluation_roc_auc(train_result_path, save_figure=True, check_point='best'):
    pred_path =os.path.join(train_result_path, 'predictions', '{}_predictions.csv'.format(check_point))
    df = pd.read_csv(pred_path)
    y_true = np.array(df.target)
    y_score = np.array(df.probability)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    if save_figure:
        plt.figure(figsize=(7, 7))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(os.path.join(train_result_path, 'predictions', '{}_eval_roc_auc.png'.format(check_point))))
        plt.close()
    return roc_auc


def evaluation_roc_auc_thresopt(train_result_path, save_figure=True, check_point='best'):
    pred_path =os.path.join(train_result_path, 'predictions',
                            '{}_predictions.csv'.format(check_point))
    df = pd.read_csv(pred_path)
    y_true = np.array(df.target)
    y_score = np.array(df.probability)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # 20210819 add G_mean to select best threshold for this binary classification task
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    gmeanOpt = round(gmean[index], ndigits=4)
    fprOpt = round(fpr[index], ndigits=4)
    tprOpt = round(tpr[index], ndigits=4)

    if save_figure:
        plt.figure(figsize=(7, 7))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve\nBest Threshold: {} with G-Mean: {}\nFPR: {}, TPR:{}'.format(thresholdOpt, gmeanOpt, fprOpt, tprOpt))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(os.path.join(train_result_path,
                                              'predictions',
                                              '{}_eval_roc_auc.png'.format(check_point))))
        plt.close()
    return roc_auc, thresholdOpt, gmeanOpt, fprOpt, tprOpt


def select_optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)

    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    gmeanOpt = round(gmean[index], ndigits=4)
    fprOpt = round(fpr[index], ndigits=4)
    tprOpt = round(tpr[index], ndigits=4)

    return thresholdOpt, gmeanOpt, fprOpt, tprOpt


def calc_err(pred, GT):
    """
    err, fpr, fnr = calc_err(pred, val_dset.targets)
    :param pred: (list) the final prediction for each slide
    :param GT: (list) ground truth, has the same length as pred
    :return: error rate, fpr, fnr
    """
    pred = np.array(pred)
    GT = np.array(GT)
    neq = np.not_equal(pred, GT)  # *Return (x1 != x2) element-wise. An array with the same shape as pred/GT
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/float(((GT==0).sum()))
    fnr = float(np.logical_and(pred==0,neq).sum())/float((GT==1).sum())
    return err, fpr, fnr
