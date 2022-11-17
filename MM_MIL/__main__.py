"""
Main function

Copyright (c) 2020 University of Illinois at Urbana-Champaign
(see LICENSE for details)
Written by Jindou Shi
"""

import os
import sys
import json
import time
import copy
import random
import numpy as np
import pandas as pd
from jsondiff import diff
import PIL.Image as Image

main_folderpath = os.path.abspath('../')
sys.path.append(main_folderpath)

# import milclassifier.train as mil_train
import MM_MIL.train as mil_train
import MM_MIL.test as mil_test
import MM_MIL.f_extractor as extractor
import MM_MIL.saliency_map as smap

mode = 0  # mode = 0: dev, mode = 1: train, mode = 2: test + saliency map, mode = 3: profiling

if mode == 0:
    # training single model
    print('************ MILclassifier ************')
    print('Copyright (c) 2020 UIUC\nWritten by Jindou Shi')
    print('***************************************')

    project_id = 'project_0_demo'

    print('Project name', project_id, '\n')
    project_path = os.path.join(main_folderpath, 'projects', project_id)
    config_path = os.path.join(project_path, 'input/config/config.json')
    with open(config_path) as f:
        config = json.load(f)
        f.close()

    mil_train.train_model(config, project_path, add_val=True, use_gpu=True, random_tile=1)
    mil_test.test_model(config, project_path, check_point='best', train_result=None)


if mode == 10:  # only test
    project_dict = {'final5_bottomK_tile_1024': ['TrainingProcess_20220129-194028']}

    for project_id in project_dict.keys():
        project_path = os.path.join(main_folderpath, 'projects', project_id)
        config_path = os.path.join(project_path, 'input/config/config.json')
        with open(config_path) as f:
            config = json.load(f)
            f.close()
        # config['dataset']['metadata']['meta_path'] = 'input/metadata/index_final_selected.csv'
        # config['dataset']['images']['image_folder'] = '/home/derek/NewStorage/All_Data/2020_WSL_SLAM_test/All_SLAM_tiles/All_images'
        # config['dataset']['images']['grid_folder'] = '/home/derek/NewStorage/All_Data/2020_WSL_SLAM_test/All_SLAM_tiles/grid_files'
        config['dataset']['images']['image_folder'] = '/home/derek/NewStorage/All_data/2020_WSL_SLAM/All_SLAM_tiles/All_images'
        config['dataset']['images']['grid_folder'] = '/home/derek/NewStorage/All_data/2020_WSL_SLAM/All_SLAM_tiles/grid_files'
        # config['dataset']['merge_object']['merge_field'] = "b++trans"
        config['dataset']['merge_object']['merge_field'] = "bag"

        for tp_id in project_dict[project_id]:
            mil_test.test_model(config, project_path, check_point='best', train_result=tp_id, only_clean_sample=False)
            mil_test.test_model(config, project_path, check_point='final', train_result=tp_id, only_clean_sample=False)

if mode == 100:  # learning rate range test
    project_id = 'project0_dev'

    print('Project name', project_id, '\n')
    project_path = os.path.join(main_folderpath, 'projects', project_id)
    config_path = os.path.join(project_path, 'input/config/config.json')
    with open(config_path) as f:
        config = json.load(f)
        f.close()

    epoch = 5
    lrs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
    values = [200, 400, 600, 800, 1000, 2000]

    config['train']['model']['epochs'] = epoch
    for lr, bs in zip(lrs, values):
        config['train']['model']['params']['learning_rate'] = lr
        config['train']['model']['params']['batch_size'] = bs
        mil_train.train_model(config, project_path, add_val=True, use_gpu=True)


elif mode == 2:  # saliency map
    print('************ MILclassifier ************')
    print('Copyright (c) 2020 UIUC\nWritten by Jindou Shi')
    print('***************************************')

    '''proj_tp_dict = {'project2_fine_tune_256': ['TrainingProcess_20211030-175627']}'''

    proj_tp_dict = {'project1_test_topK_256': ['TrainingProcess_20211024-005744']}

    for project_id in proj_tp_dict.keys():
        t_results = proj_tp_dict[project_id]
        print('Project name', project_id, '\n')
        project_path = os.path.join(main_folderpath, 'projects', project_id)

        for tp_id in t_results:
            config_path = os.path.join(project_path, 'output', tp_id, 'config', 'config_copy.json')
            with open(config_path) as f:
                config = json.load(f)
                f.close()
            smap.saliency_map(config, project_path, check_point='best', train_result=tp_id,
                              only_clean_sample=False)
