#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')  # TODO: Make this relative path maybe
from evaluation.evaluation import get_pooled_aggregate_stats
from sklearn.model_selection import ParameterGrid

# To reduce randomness
import torch 
import numpy as np
SEED = 13
torch.manual_seed(SEED)
np.random.seed(SEED)


RANKING_OBJECTS_DIR = r'/home/scratch/mgoswami/Experiments_Oct29/results/'  # Directory where all ranking objects are saved
SAVE_DIR = r'/home/scratch/mgoswami/Experiments_VUS'

evaluation_params = [
    {
        'save_dir': [RANKING_OBJECTS_DIR],
        'dataset': ['anomaly_archive'],
        'data_family': [
            'Atrial Blood Pressure (ABP)',
            'Electrocardiogram (ECG) Arrhythmia',
            'Insect Electrical Penetration Graph (EPG)',
            'Power Demand',
            'NASA Data',
            'Gait',
            'Respiration Rate (RESP)',
            'Acceleration Sensor Data',
            'Air Temperature',
        ],
        'evaluation_metric': ['VUS'],
        'n_validation_splits': [5],
        'n_neighbors': [[2, 4, 6]],
        'random_state': [13],
        'n_splits': [100],
        'metric': ['influence'],
        'use_all_ranks': [False],
        'top_k': [3],
        'top_kr': [None],
        'n_jobs': [5]
    }
]

# NOTE: Not conducting experiments on SMD because it is multivariate and VUS does not support that (to the best of my knowledge)

aggregate_stats = {}
for params in list(ParameterGrid(evaluation_params)):
    stats = get_pooled_aggregate_stats(**params)

    if 'data_family' in params.keys():
        aggregate_stats[params['data_family']] = stats
    else:
        aggregate_stats['smd'] = stats

    with open(os.path.join(SAVE_DIR, f"aggregate_stats_{params['dataset']}.pkl"), 'wb') as f:
        pkl.dump(aggregate_stats, f)
