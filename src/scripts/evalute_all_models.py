#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This script generates the ranking objects for all the datasets
"""

import numpy as np
import os
import pickle
import pandas as pd
import sys
from joblib import Parallel, delayed

print('Loaded 1!')
sys.path.append('/zfsauton2/home/mgoswami/tsad-model-selection/src/')

from model_selection.model_selection import RankModels
from model_selection.utils import Logger
from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES

print('Loaded 2!')

# To reduce randomness
import torch 
import numpy as np
SEED = 13
torch.manual_seed(SEED)
np.random.seed(SEED)

# DATASETS = ['anomaly_archive', 'smd']
# ENTITIES = [MACHINES, ANOMALY_ARCHIVE_ENTITIES]
# DATASETS = ['anomaly_archive']
# ENTITIES = [ANOMALY_ARCHIVE_ENTITIES[:20]]
DATASETS = ['smd']
# ENTITIES = [MACHINES]
ENTITIES = [['machine-1-1']]

RESULTS_PATH = r'/home/scratch/mgoswami/results/'  # Directory to save the results
TRAINED_MODEL_PATH = r'/home/scratch/mgoswami/trained_models/'
DATASET_PATH = r'/home/scratch/mgoswami/datasets/'
N_JOBS = 1  # Number of parallel jobs to run

# Logging object
VERBOSE = False
OVERWRITE = True
DOWNSAMPLING = 10  # None

# Logger object to save the models
logging_obj = Logger(save_dir=RESULTS_PATH,
                     overwrite=OVERWRITE,
                     verbose=VERBOSE)

### Parameters for our Ranking Model
rank_model_params = {
    'trained_model_path': TRAINED_MODEL_PATH,
    'min_length': 256,
    'root_dir': DATASET_PATH,
    'normalize': True,
    'verbose': False
}

### Parameters for our Evaluation Model
evaluate_model_params = {
    'n_repeats': 2,
    'n_neighbors': [4, 10, 16],
    'split': 'test',
    'synthetic_ranking_criterion': 'f1',  # Can be one of f1 or prauc
    'n_splits': 100,
}

unevaluated_entities = []  # List of entities which remain unevaluated


def evaluate_model_wrapper(dataset, entity):
    print(42 * "=")
    print(f"Evaluating models on entity: {entity}")
    print(42 * "=")

    rank_model_params['dataset'] = dataset
    rank_model_params['entity'] = entity
    rank_model_params['downsampling'] = DOWNSAMPLING
    logging_hierarchy = [dataset]

    if not OVERWRITE:
        if logging_obj.check_file_exists(obj_class=logging_hierarchy,
                                         obj_name=f'ranking_obj_{entity}'):
            print(f'Models on entity {entity} already evaluated!')
            return

    # Create a ranking object
    ranking_obj = RankModels(**rank_model_params)

    # try:
    _ = ranking_obj.evaluate_models(**evaluate_model_params)
        # _ = ranking_obj.rank_models()
    # except:
        # unevaluated_entities.append((dataset, entity))
        # return

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
