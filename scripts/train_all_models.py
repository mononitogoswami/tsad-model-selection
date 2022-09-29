#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to pre-train models on all the datasets/entities
#######################################

import sys
import traceback

sys.path.append(
    '/home/ubuntu/TSADModelSelection/')  # TODO: Make this relative path maybe
from model_trainer.trainer import TrainModels
from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, ANOMALY_ARCHIVE_10_ENTITIES, MACHINES, MSL_CHANNELS, SMAP_CHANNELS
from joblib import Parallel, delayed
import numpy as np

DATASETS = ['anomaly_archive', 'smd', 'msl', 'smap']
ENTITIES = [ANOMALY_ARCHIVE_ENTITIES, MACHINES, MSL_CHANNELS, SMAP_CHANNELS]

N_JOBS = 4


def train_model_wrapper(dataset, entities):
    print(42 * "=")
    print(f"Training models on entity: {entities}")
    print(42 * "=")
    model_trainer = TrainModels(
        dataset=dataset,
        entity=entities,
        downsampling=10 if dataset in ['anomaly_archive', 'smd'] else None,
        min_length=256,
        root_dir='/home/ubuntu/datasets/',
        training_size=1,
        overwrite=False,
        verbose=True,
        save_dir='/home/ubuntu/efs/trained_models')

    try:
        model_trainer.train_models(model_architectures='all')
    except:  # Handle exceptions to allow continue training
        print(f'Traceback for Entity: {entities} Dataset: {dataset}')
        print(traceback.format_exc())
    print(42 * "=")


for d_i, dataset in enumerate(DATASETS):
    _ = Parallel(n_jobs=N_JOBS)(delayed(train_model_wrapper)(dataset, entities)
                                for entities in ENTITIES[d_i])
