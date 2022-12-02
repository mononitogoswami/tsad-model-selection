#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to pre-train models on all the datasets/entities
#######################################

import traceback
import sys

sys.path.append('../')  # TODO: Make this relative path maybe
from model_trainer.trainer import TrainModels
from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES
from joblib import Parallel, delayed
import numpy as ndp

# To reduce randomness
import torch 
import numpy as np
SEED = 13
torch.manual_seed(SEED)
np.random.seed(SEED)

# DATASETS = ['anomaly_archive', 'smd']
# ENTITIES = [MACHINES, ANOMALY_ARCHIVE_ENTITIES]
# DATASETS = ['anomaly_archive']
# ENTITIES = [ANOMALY_ARCHIVE_ENTITIES[:10]]
DATASETS = ['smd']
ENTITIES = [MACHINES[:7]]

N_JOBS = 2
DOWNSAMPLING = 10  # None
# Directory to save the trained models
TRAINED_MODEL_PATH = '/home/scratch/mgoswami/trained_models'  # '/home/scratch/mgoswami/trained_models_wo_downsampling'
DATASET_PATH = '/home/scratch/mgoswami/datasets/'


def train_model_wrapper(dataset, entities):
    print(42 * "=")
    print(f"Training models on entity: {entities}")
    print(42 * "=")
    model_trainer = TrainModels(dataset=dataset,
                                entity=entities,
                                downsampling=DOWNSAMPLING,
                                min_length=256,
                                root_dir=DATASET_PATH,
                                training_size=1,
                                overwrite=False,
                                verbose=True,
                                save_dir=TRAINED_MODEL_PATH)

    try:
        model_trainer.train_models(model_architectures='all')
    except:  # Handle exceptions to allow continue training
        print(f'Traceback for Entity: {entities} Dataset: {dataset}')
        print(traceback.format_exc())
    print(42 * "=")


for d_i, dataset in enumerate(DATASETS):
    _ = Parallel(n_jobs=N_JOBS)(delayed(train_model_wrapper)(dataset, entities)
                                for entities in ENTITIES[d_i])
