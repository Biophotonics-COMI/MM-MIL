"""
Training

Copyright (c) 2020 University of Illinois at Urbana-Champaign
(see LICENSE for details)
Written by Jindou Shi
"""

'''
To-do:
0. config.json [done]
1. change input size, customize input layer of the model
2. normalization function
3. customize loss function (criterion)
4. torch -> tensor

Future:
1. multi-class
2. image processing (illumination correction, compression/reshape)
3. 
'''

import os
import sys
import json
import copy
import time
import copy
import random
import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
import PIL.Image as Image
from jsondiff import diff
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
from . import dataset as mil_data
from . import utils as mil_utils
from . import evaluation as mileval


def train_model(config, project_path, add_val=True, use_gpu=True, random_tile=0):
    """
    Training MIL models
    :param config: (dict) configurations read from config.json
    :param project_path: (str) folder path of this project, which contain 'input' and 'output' folder
    :param add_val: (bool) 
    :param use_gpu: (bool) 
    """
    print('\n********** Train mode **********\n')
    '''load configurations'''
    meta_path = os.path.join(os.path.abspath(project_path), config['dataset']['metadata']['meta_path'])
    merge_flag = config["dataset"]["merge_object"]["flag"]

    nepochs = config['train']['model']['epochs']
    batch_size = config['train']['model']['params']['batch_size']
    lr = config['train']['model']['params']["learning_rate"]
    weight_decay = config['train']['model']['params']["weight_decay"]
    k = config['train']['model']['top_k']
    pos_weights = config['train']['criterion']['positive_class_weights']
    workers = config['train']['load_worker']
    test_every = config['train']['validation']['test_every']
    add_val = config['train']['validation']['add_val']
    mean_list = config['prepare']['normalization']['mean']
    std_list = config['prepare']['normalization']['std']

    model_name = config['train']['model']['name']
    model_momentum = config['train']['model']['momentum']
    model_lite = config['train']['model']['lite']
    model_pretrain = config['train']['model']['pretrained']
    c_count = len(config["dataset"]["images"]["channels"])
    fix_weight = config["train"]["model"]["fix_weight"]
    model_path = os.path.join(os.path.abspath(project_path), config["train"]["model"]["pretrained"])
    class_num = config["train"]["target"]["class_count"]

    save_folder = '/home/derek/All_Data/2020_WSL_SLAM/SSL_SLAM_all_tiles/All_images'
    grid_folder = '/home/derek/All_Data/2020_WSL_SLAM/SSL_SLAM_all_tiles/grid_files'
    pretrain_folder = '/home/derek/DerekProjects/BIL_projects/2020_Weakly-Supervised-Learning_SLAM/WSL_SLAM_2021_final_new_bag/assets/weights'

    min_epoch = 50

    '''define normalization'''
    # normalize, trans, mean_list, std_list = normalization(config, project_path)

    '''load data'''
    train_dset = mil_data.MILdataset(meta_path, config, 'train', transform=None)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False)
    if add_val:
        val_dset = mil_data.MILdataset(meta_path, config, 'val', transform=None)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=False)

    print('=' * 100 + '\nConfigurations:\n')
    pprint.pprint(config)
    print('=' * 100)

    '''create output file'''
    output_path = create_output_folders(config, project_path)

    '''model'''
    print("\nModel: {} Pretrained: {} Fixed: {}".format(model_name, model_pretrain, fix_weight))
    model = mil_model.resnet34()

    # loading pretrained model
    if model_pretrain != '':
        ch = torch.load(os.path.join(pretrain_folder, model_pretrain))
        pretrain_dict = fix_state_dict(ch['model'])
        model_dict = model.state_dict()
        temp_len = len(pretrain_dict)
        pretrain_dict = {key: v for key, v in pretrain_dict.items() if key in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print('Loading pretrained model: {}\t{}/{} layers '.format(model_pretrain, len(pretrain_dict), temp_len))

    # Multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()), end='\n\n')
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()

    '''criterion'''
    if pos_weights == 0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1 - pos_weights, pos_weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 2022 0110 changed
    # rate_positive = sum(train_dset.object_targets) / float(len(train_dset.object_targets))
    '''rate_positive = train_dset.positive_rate
    r_negative = rate_positive * (1 - pos_weights)
    r_positive = (1 - rate_positive) * pos_weights
    w = torch.Tensor([r_negative / (r_negative + r_positive), r_positive / (r_negative + r_positive)])
    criterion = nn.CrossEntropyLoss(w).cuda()
    print('\nPositive class ratio: {}\tPositve class weight: {}\nLoss class weight: {}\n'.format(rate_positive, pos_weights, w))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)'''

    '''cudnn.benchmark'''
    cudnn.benchmark = True

    '''loop through epochs'''
    '''previous_model = copy.deepcopy(model)'''
    best_acc = 0
    # best_auc = 0.
    for epoch in range(nepochs):
        time_0 = time.time()
        train_dset.setmode(1)
        print('Epoch: [{}/{}]\tInference/Training\tBatch processing:'.format(epoch + 1, nepochs))
        time_1 = time.time()
        probs = inference(config, train_loader, model)
        time_2 = time.time()
        # topk = group_argtopk_add_random_tile(np.array(train_dset.objectID), probs, k, random_tile)
        topk, bottom_k = group_argtopk_neg_tile(np.array(train_dset.objectID), probs, k, k)

        time_3 = time.time()
        train_dset.maketraindata(topk, bottom_k)
        time_4 = time.time()
        train_dset.shuffletraindata()
        time_5 = time.time()
        train_dset.setmode(2)
        time_6 = time.time()
        loss = train(train_loader, model, criterion, optimizer)  # training process
        time_7 = time.time()

        # Momentum update
        '''for p1, p2 in zip(model.parameters(), previous_model.parameters()):
            p1.data = p2.data * model_momentum + p1.data * (1. - model_momentum)
        previous_model = copy.deepcopy(model)'''

        print('Training loss: {}\tTopK: {}'.format(loss, len(train_dset.t_data)))

        t1 = time_1 - time_0
        t2 = time_2 - time_1
        t3 = time_3 - time_2
        t4 = time_4 - time_3
        t5 = time_5 - time_4
        t6 = time_6 - time_5
        t7 = time_7 - time_6
        t_all = time_7 - time_0

        print('-> Total time: {}'.format(t_all))

        print(
            '-> Time: 1:{:.4f}, 2:{:.4f}, 3:{:.4f}, 4:{:.4f}, 5:{:.4f}, 6:{:.4f}, 7:{:.4f}'.format(t1, t2, t3, t4, t5, t6,
                                                                                                t7))
        print('-> Time_perc: 1:{:.3f}, 2:{:.3f}, 3:{:.3f}, 4:{:.3f}, 5:{:.3f}, 6:{:.3f}, 7:{:.3f}'.format(t1 / t_all,
                                                                                                       t2 / t_all,
                                                                                                       t3 / t_all,
                                                                                                       t4 / t_all,
                                                                                                       t5 / t_all,
                                                                                                       t6 / t_all,
                                                                                                       t7 / t_all))

        # Validation
        if add_val and (epoch + 1) % test_every == 0:
            print('='*100)
            print('Epoch: [{}/{}]\tValidation\tBatch processing:'.format(epoch + 1, nepochs))
            val_dset.setmode(3)
            # probs = inference(config, val_loader, model)
            probs, loss_val = inference_val(config, val_loader, model, criterion)
            if merge_flag:
                maxs = group_max(np.array(val_dset.objectID), probs, max(val_dset.objectID)+1)  # probs for each object
                pred = [1 if x >= 0.5 else 0 for x in maxs]
                print('predictions: {}'.format(pred))
                print('    targets: {}'.format(val_dset.object_targets))
                err, fpr1, fnr1 = calc_err(pred, val_dset.object_targets)
                roc_auc, thres, gmean, fpr, fnr, acc = mileval.validation_roc_auc_thresopt(val_dset.object_targets, maxs)

            else:
                maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))  # probs for each slide
                pred = [1 if x >= 0.5 else 0 for x in maxs]
                err, fpr1, fnr1 = calc_err(pred, val_dset.targets)
                roc_auc, thres, gmean, fpr, fnr, acc = mileval.validation_roc_auc_thresopt(val_dset.targets, maxs)

            print('Validation loss: {}\tACC: {:.4f}\tFPR: {:.4f}\tFNR: {:.4f}'.format(loss_val, 1 - (fpr1 + fnr1) / 2., fpr1, fnr1))
            print('ROC-AUC results: AUC: {:.4f}\tACC: {:.4f}\tFPR: {:.4f}\tFNR: {:.4f}\tOptThres: {:.4f}'.format(roc_auc, acc,
                                                                                                 fpr, fnr, thres))
            print('=' * 100)
            fconv = open(os.path.join(output_path, 'logs', 'convergence.csv'), 'a')
            fconv.write('{},{},{},{},{},{}\n'.format(epoch+1, loss, loss_val, 1-err, fpr1, fnr1))
            fconv.close()

            # Save best model
            err = (fpr1 + fnr1) / 2.
            if (1 - err >= best_acc) and epoch + 1 >= min_epoch:
                best_acc = 1 - err
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'best_auc': roc_auc,
                    'optimizer': optimizer.state_dict(),
                    'normalization': (mean_list, std_list)
                }
                torch.save(obj, os.path.join(output_path, 'weights', 'checkpoint_best.pth'))

            # save last model
            if epoch == nepochs-1:
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': 1 - err,
                    'best_auc': roc_auc,
                    'optimizer': optimizer.state_dict(),
                    'normalization': (mean_list, std_list)
                }
                torch.save(obj, os.path.join(output_path, 'weights', 'checkpoint_final.pth'))
        else:
            fconv = open(os.path.join(output_path, 'logs', 'convergence.csv'), 'a')
            fconv.write('{},{},{},{},{},{}\n'.format(epoch+1, loss, float('nan'), float('nan'), float('nan'), float('nan')))
            fconv.close()
    mil_utils.export_train_report(os.path.join(output_path, 'logs'))
    print('\n=========== Done! ===========\n')


def inference(config, loader, model):
    """inference the model using loader dataset, generate probability for each tile
    :param config: (dict) configuration info read from config.json
    :param loader: (torch.DataLoader) each element is one batch
    :param model: (torchvision.models) model
    :return probs: (numpy array) the probability of each tile
    """
    batch_size = config['train']['model']['params']['batch_size']
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(tqdm(loader, ncols=100)):
            input = input.float()
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*batch_size:i*batch_size+input.size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def inference_val(config, loader, model, criterion):
    """inference the model using loader dataset, generate probability for each tile
    :param config: (dict) configuration info read from config.json
    :param loader: (torch.DataLoader) each element is one batch
    :param model: (torchvision.models) model
    :param criterion: (torch.nn.CrossEntropyLoss) the loss function
    :return probs: (numpy array) the probability of each tile
    """
    batch_size = config['train']['model']['params']['batch_size']
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    running_loss = 0.
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(loader, ncols=100)):
            input = input.float()
            input = input.cuda()
            target = target.cuda()
            o = model(input)
            output = F.softmax(o, dim=1)
            loss = criterion(o, target)
            running_loss += loss.item() * input.size(0)
            probs[i*batch_size:i*batch_size+input.size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy(), running_loss/len(loader.dataset)


def train(loader, model, criterion, optimizer):
    """ Training the model
    :param loader: (torch.DataLoader) each element is one batch
    :param model: (torchvision.models) model object
    :param criterion: (torch.nn.CrossEntropyLoss) the loss function
    :param optimizer: (torch.optim) optimization method
    :return: (float) averaged loss for each tile
    """
    model.train()  # set tp train mode, set model.training = True
    running_loss = 0.
    for i, (input, target) in enumerate(tqdm(loader, ncols=100)):
        input = input.float()
        input = input.cuda()
        target = target.cuda()
        output = model(input)   # output shape: (num_tile_in_this_batch, num_class)
        loss = criterion(output, target)   # calculate loss function, with gradients. One loss value for one batch (averaged)
        optimizer.zero_grad()   # Clears the gradients of all optimized Tensors
        loss.backward()   # computes dloss/dx for every parameter x which has requires_grad=True
        optimizer.step()   # updates the parameters. apply after gradients are computed (e.g. .backward())
        running_loss += loss.item()*input.size(0)  # accumulating total loss of this batch
    return running_loss/len(loader.dataset)  # averaged loss for each tile


def normalization(config, project_path):
    """Define normalization method
    :param config: (dict) configuration info read from config.json
    :param project_path: (str) folder path of this project, which contain 'input' and 'output' folder
    :return: normalize, trans
    """
    meta_path = os.path.join(os.path.abspath(project_path), config['dataset']['metadata']['meta_path'])
    norm_fix_value_flag = config['prepare']['normalization']['apply_fixed_value']
    norm_mean = config['prepare']['normalization']['mean']
    norm_std = config['prepare']['normalization']['std']
    c_count = len(config['dataset']['images']['channels'])
    if norm_fix_value_flag:
        mean_list = norm_mean
        std_list = norm_std
    else:
        mean_list, std_list = mil_utils.generate_stats_report(config, meta_path, save_flag=True, norm_flag=True)
    assert len(mean_list) == c_count and len(std_list) == c_count, '[Error] image stats not matching channel count'
    normalize = transforms.Normalize(mean=mean_list, std=std_list)
    trans = transforms.Compose([transforms.ToTensor(),
                                normalize])
    return normalize, trans, mean_list, std_list


def create_output_folders(config, project_path):
    """Create output folders
    :param config: (dict) configurations read from config.json
    :param project_path: (str) folder path of this project, which contain 'input' and 'output' folder
    :return output_path: (str) folder path for outputs
    """
    sub_folders = ['weights', 'config', 'logs', 'predictions']
    timestr = time.strftime(r"%Y%m%d-%H%M%S")
    output_path = os.path.join(project_path, 'output', 'TrainingProcess_' + timestr)
    print('\nCreating project output folder: {}'.format(output_path))
    for f in sub_folders:
        os.makedirs(os.path.join(output_path, f))
    with open(os.path.join(output_path, 'config', 'config_copy.json'), 'w') as json_file:
        json.dump(config, json_file, indent=4)
        json_file.close()

    # config comparison (what is modified)
    ori_path = os.path.abspath('../assets/config_original.json')
    with open(ori_path) as f:
        config_ori = json.load(f)
        f.close()
    diff_dict = diff(config_ori, config)
    with open(os.path.join(output_path, 'config', 'config_modification.json'), 'w') as json_file:
        json.dump(diff_dict, json_file, indent=4)
        json_file.close()

    fconv = open(os.path.join(output_path, 'logs', 'convergence.csv'), 'w')
    fconv.write('epoch,train_loss,val_loss,val_acc,val_fpr,val_fnr\n')
    fconv.close()
    return output_path


def calc_err(pred,real):
    """
    err, fpr, fnr = calc_err(pred, val_dset.targets)
    :param pred: (list) the final prediction for each slide
    :param real: (list) ground truth, has the same length as pred
    :return: error rate, fpr, fnr
    """
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)  # *Return (x1 != x2) element-wise. An array with the same shape as pred/real
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr


def group_argtopk(groups, data, k=1):
    """ Return the index of top k tiles in each slide
    :param groups: (1d array) an array of dset.slideIDX, with shape (total_tile_count,)
    :param data: (1d array) an array of probabilities, with same shape as groups
    :param k: (int) top k tiles
    :return: (list) 
    """
    order = np.lexsort((data, groups))   # **sort groups based on: 1.slideIDX, 2.prob  --> order
    groups = groups[order]   # *reorder the slideIDX, the last one has the highest probability!
    # data = data[order]
    index = np.empty(len(groups), 'bool')  # a bool array, len = total_tile_count
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]   # affect the rest, (len(index)-k) elements. 
    return list(order[index])  # select index, len(output_list) = k * slide_count


def group_argtopk_add_random_tile(groups, data, k=1, random=0):
    """ Return the index of top k tiles in each slide
    :param groups: (1d array) an array of dset.slideIDX, with shape (total_tile_count,)
    :param data: (1d array) an array of probabilities, with same shape as groups
    :param k: (int) top k tiles
    :return: (list)
    """

    order = np.lexsort((data, groups))   # **sort groups based on: 1.slideIDX, 2.prob  --> order
    groups = groups[order]   # *reorder the slideIDX, the last one has the highest probability!
    index = np.empty(len(groups), 'bool')  # a bool array, len = total_tile_count
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]   # affect the rest, (len(index)-k) elements.

    if random != 0:
        # focus on the unselected tiles
        num_object = np.max(groups) + 1
        num_random_tile = num_object * random * k
        un_index = np.where(index != True)[0]
        temp_index = np.random.choice(un_index, num_random_tile)
        index[temp_index] = True
    return list(order[index])  # select index, len(output_list) = k * slide_count


def group_argtopk_neg_tile(groups, data, k=1, neg_k=0):
    """ Return the index of top k tiles in each slide
    :param groups: (1d array) an array of dset.slideIDX, with shape (total_tile_count,)
    :param data: (1d array) an array of probabilities, with same shape as groups
    :param k: (int) top k tiles
    :return: (list)
    """
    # top K instances
    order = np.lexsort((data, groups))   # **sort groups based on: 1.slideIDX, 2.prob  --> order
    groups = groups[order]   # *reorder the slideIDX, the last one has the highest probability!
    index = np.empty(len(groups), 'bool')  # a bool array, len = total_tile_count
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]   # affect the rest, (len(index)-k) elements.

    # negative instances from positive bags
    if neg_k != 0:
        # focus on the unselected tiles
        index_neg = np.empty(len(groups), 'bool')
        index_neg[:k] = True
        index_neg[k:] = groups[:-k] != groups[k:]

    return list(order[index]), list(order[index_neg])  # select index, len(output_list) = k * slide_count


def group_max(groups, data, nmax):
    """Generate a list of probs for slides, one prob for one slide/object
    :param groups: (1d array)  an array of dset.slideIDX, with shape (total_tile_count,)
    :param data: (1d array) an array of probabilities, with same shape as groups
    :param nmax: (int) target number,  equals slide/object number
    :return:
    """
    out = np.empty(nmax)   # shape (slide_num)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]  # only one element remains for each slide
    return list(out)   # [slide_1_prob, slide_2_prob, ...]


def fix_state_dict(state_dict):
    """remove .module from keys, caused by nn.DataParallel"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[13:]  # remove `module.model.`
        new_state_dict[name] = v
    return new_state_dict
































