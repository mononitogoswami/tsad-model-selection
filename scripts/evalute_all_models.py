#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pickle
import pandas as pd
import sys

sys.path.append('/home/ubuntu/PyMAD/')  # TODO: Make this relative path maybe
sys.path.append('/home/ubuntu/TSADModelSelection')

from model_selection.model_selection import RankModels
from model_selection.utils import Logger
from model_selection.utils import visualize_predictions, visualize_data
from model_selection.rank_aggregation import trimmed_borda, trimmed_kemeny, borda, kemeny
from metrics.metrics import evaluate_model_selection
from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES, MSL_CHANNELS, SMAP_CHANNELS
from joblib import Parallel, delayed

DATASETS = ['smd', 'anomaly_archive']  #, 'msl', 'smap']
ENTITIES = [MACHINES,
            ANOMALY_ARCHIVE_ENTITIES]  #, MSL_CHANNELS, SMAP_CHANNELS]

SAVE_DIR = '/home/ubuntu/efs/results'  # Directory to save the results
# TOP_K = 15 # Number of metrics to choose for trimmed rank aggregation
# k = 5 # Top k models for evaluation
N_JOBS = 2  # Number of parallel jobs to run

# Logging object
VERBOSE = False
OVERWRITE = True

# Logger object to save the models
logging_obj = Logger(save_dir=SAVE_DIR, overwrite=OVERWRITE, verbose=VERBOSE)

### Parameters for our Ranking Model
rank_model_params = {
    'trained_model_path': r'/home/ubuntu/efs/trained_models/',
    'min_length': 256,
    'root_dir': '/home/ubuntu/datasets/',
    'normalize': True,
    'verbose': False
}

### Parameters for our Evaluation Model
evaluate_model_params = {
    'n_repeats': 2,
    'n_neighbors': [4, 10, 16],
    'split': 'test',
    'synthetic_ranking_criterion': 'prauc',
    'n_splits': 100,
}

unevaluated_entities = []  # List of entities which remain unevaluated


def evaluate_model_wrapper(dataset, entity):
    print(42 * "=")
    print(f"Evaluating models on entity: {entity}")
    print(42 * "=")

    rank_model_params['dataset'] = dataset
    rank_model_params['entity'] = entity
    rank_model_params['downsampling'] = 10 if dataset in [
        'anomaly_archive', 'smd'
    ] else None
    logging_hierarchy = [dataset]

    if not OVERWRITE:
        if logging_obj.check_file_exists(obj_class=logging_hierarchy,
                                         obj_name=f'ranking_obj_{entity}'):
            print(f'Models on entity {entity} already evaluated!')
            return

    # Create a ranking object
    ranking_obj = RankModels(**rank_model_params)

    try:
        _ = ranking_obj.evaluate_models(**evaluate_model_params)
        _ = ranking_obj.rank_models()
    except:
        unevaluated_entities.append((dataset, entity))
        return

    # Save the ranking objection for later use
    logging_obj.save(obj=ranking_obj,
                     obj_name=f'ranking_obj_{entity}',
                     obj_meta=None,
                     obj_class=logging_hierarchy,
                     type='data')


for d_i, dataset in enumerate(DATASETS):
    _ = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_model_wrapper)(dataset, entities)
        for entities in ENTITIES[d_i])

# Log unevaluated entities
logging_obj.save(obj=unevaluated_entities,
                 obj_name=f'unevaluated_entities',
                 obj_meta=None,
                 obj_class=[],
                 type='data')
