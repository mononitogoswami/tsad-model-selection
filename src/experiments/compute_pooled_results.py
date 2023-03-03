#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to evaluate model selection performance
#######################################

import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.model_selection import ParameterGrid

from tsadams.utils.config import Config
from tsadams.utils.set_all_seeds import set_all_seeds
from tsadams.evaluation.evaluation import get_pooled_aggregate_stats

def set_eval_params(args):
    evaluation_params = [
        {
            'save_dir': [args['results_path']],
            'dataset': ['smd'],
            'data_family': ['SMD'],
            'evaluation_metric': [args['evaluation_metric']],
            'n_validation_splits': [args['n_validation_splits']],
            'n_neighbors': [args['n_neighbors']],
            'random_state': [args['random_state']],
            'n_splits': [args['n_splits']],
            'sliding_window': [args['sliding_window']],
            'metric': [args['metric']],
            'top_k': [args['top_k']],
            'use_all_ranks': [args['use_all_ranks']],
            'top_kr': [args['top_kr']],
            'n_jobs': [args['n_jobs']]
        },
        {
            'save_dir': [args['results_path']],
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
            'evaluation_metric': [args['evaluation_metric']],
            'n_validation_splits': [args['n_validation_splits']],
            'n_neighbors': [args['n_neighbors']],
            'random_state': [args['random_state']],
            'n_splits': [args['n_splits']],
            'sliding_window': [args['sliding_window']],
            'metric': [args['metric']],
            'top_k': [args['top_k']],
            'use_all_ranks': [args['use_all_ranks']],
            'top_kr': [args['top_kr']],
            'n_jobs': [args['n_jobs']]
        },
    ]
    return evaluation_params

def main():
    parser = ArgumentParser(description='Config file')
    parser.add_argument('--config_file_path',
                        '-c', 
                        type=str, 
                        default='config.yaml',
                        help='path to config file')
    args = parser.parse_args()
    args = Config(config_file_path=args.config_file_path).parse()
   
    set_all_seeds(args['random_seed']) # Reduce randomness

    evaluation_params = set_eval_params(args)

    aggregate_stats = {}
    for params in list(ParameterGrid(evaluation_params)):
        stats = get_pooled_aggregate_stats(**params)

        if 'data_family' in params.keys():
            aggregate_stats[params['data_family']] = stats
        else:
            aggregate_stats['smd'] = stats

        with open(os.path.join(args['results_path'], f"aggregate_stats_{params['dataset']}.pkl"), 'wb') as f:
            pkl.dump(aggregate_stats, f)

if __name__ == '__main__':
    main()    
