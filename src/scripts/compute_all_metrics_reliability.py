#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Experiment: What happens when all metrics are used? 

import pandas as pd
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')  # TODO: Make this relative path maybe
from evaluation.evaluation import get_pooled_reliability
from sklearn.model_selection import ParameterGrid

# To reduce randomness
import torch 
import numpy as np
SEED = 13
torch.manual_seed(SEED)
np.random.seed(SEED)

RANKING_OBJECTS_DIR = r'/home/scratch/mgoswami/Experiments_Oct29/results/'  # Directory where all ranking objects are saved
SAVE_DIR = r'/home/scratch/mgoswami/Experiments_reliability'

evaluation_params = [
    {
        'save_dir': [RANKING_OBJECTS_DIR],
        'dataset': ['smd'],
        'data_family': ['SMD'],
        'evaluation_metric': ['Best F-1'],
        'n_neighbors': [[2, 4, 6]],
        'random_state': [13],
        'n_splits': [100],
        'sliding_window': [None],
        'metric': ['influence'],
        'top_k': [3],
        'n_jobs': [5]
    },
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
        'evaluation_metric': ['Best F-1'],
        'n_neighbors': [[2, 4, 6]],
        'random_state': [13],
        'n_splits': [100],
        'sliding_window': [None],
        'metric': ['influence'],
        'top_k': [3],
        'n_jobs': [5]
    },
]

aggregate_stats = {}
for params in list(ParameterGrid(evaluation_params)):
    stats = get_pooled_reliability(**params)

    if 'data_family' in params.keys():
        aggregate_stats[params['data_family']] = stats
    else:
        aggregate_stats['smd'] = stats

    with open(os.path.join(SAVE_DIR, f"aggregate_stats_{params['dataset']}.pkl"), 'wb') as f:
        pkl.dump(aggregate_stats, f)
