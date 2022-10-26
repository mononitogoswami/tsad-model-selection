#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

######################################################
# Function to check the number of models already trained
######################################################

import os
import sys

sys.path.append('../')  # TODO: Make this relative path maybe
from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES

DATASETS = ['anomaly_archive', 'smd']
ENTITIES = [ANOMALY_ARCHIVE_ENTITIES, MACHINES]
# TRAINED_MODEL_BASE_PATH = r'/home/scratch/mgoswami/trained_models/'
TRAINED_MODEL_BASE_PATH = r'/home/scratch/mgoswami/trained_models_wo_downsampling/'


def main():
    total_models = 0
    for d, dataset in enumerate(DATASETS):
        total_models_per_dataset = 0
        if not os.path.exists(os.path.join(TRAINED_MODEL_BASE_PATH, dataset)):
            print(f'No models trained for dataset {dataset}')
        else:
            for entity in ENTITIES[d]:
                if not os.path.exists(
                        os.path.join(TRAINED_MODEL_BASE_PATH, dataset,
                                     entity)):
                    print(
                        f'No models trained for entity {entity} of dataset {dataset}'
                    )
                else:
                    n_trained_models = int(
                        len(
                            os.listdir(
                                os.path.join(TRAINED_MODEL_BASE_PATH, dataset,
                                             entity))) / 2)
                    print(
                        f"Entity: {entity} | Number of models trained: {n_trained_models}"
                    )
                    total_models_per_dataset = total_models_per_dataset + n_trained_models

        total_models = total_models + total_models_per_dataset
        print(
            f'Total models trained in {dataset} dataset: {dataset} = {total_models_per_dataset}'
        )
    print(f'Total number of models trained = {total_models}')


if __name__ == '__main__':
    main()
