#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

######################################################
# Function to check the number of evaluated entities
######################################################

import os
import sys

sys.path.append('../')  # TODO: Make this relative path maybe
from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES

DATASETS = ['anomaly_archive', 'smd']
ENTITIES = [ANOMALY_ARCHIVE_ENTITIES, MACHINES]
EVALUATED_MODEL_BASE_PATH = r'/home/scratch/mgoswami/results/'

total_models = 0
for d, dataset in enumerate(DATASETS):
    n_evaluated_models = 0
    if not os.path.exists(os.path.join(EVALUATED_MODEL_BASE_PATH, dataset)):
        print(f'No models evaluated for dataset {dataset}')
    else:
        n_evaluated_models = int(
            len(os.listdir(os.path.join(EVALUATED_MODEL_BASE_PATH, dataset))))
        print(f'Total entities evaluated in {dataset} = {n_evaluated_models}')

    total_models = total_models + n_evaluated_models
print(f'Total number of entities evaluated = {total_models}')
