#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to train models on all the datasets/entities
#######################################

import traceback
from joblib import Parallel, delayed
from tsadams.model_trainer.train import TrainModels
from tsadams.model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES
from tsadams.utils.utils import get_args_from_cmdline
from tsadams.utils.set_all_seeds import set_all_seeds

def train_model_wrapper(dataset, entities, args):
    print(42 * "=")
    print(f"Training models on entity: {entities}")
    print(42 * "=")
    model_trainer = TrainModels(dataset=dataset,
                                entity=entities,
                                downsampling=args['downsampling'],
                                min_length=args['min_length'],
                                root_dir=args['dataset_path'],
                                training_size=args['training_size'],
                                overwrite=args['overwrite'],
                                verbose=args['verbose'],
                                save_dir=args['trained_model_path'])
    try:
        model_trainer.train_models(model_architectures=args['model_architectures'])
    except:  # Handle exceptions to allow continue training
        print(f'Traceback for Entity: {entities} Dataset: {dataset}')
        print(traceback.format_exc())
    print(42 * "=")

def main():
    args = get_args_from_cmdline()
   
    print('Set all seeds!')
    set_all_seeds(args['random_seed']) # Reduce randomness
    
    DATASETS = ['anomaly_archive', 'smd']
    ENTITIES = [ANOMALY_ARCHIVE_ENTITIES, MACHINES]

    for d_i, dataset in enumerate(DATASETS):
        _ = Parallel(n_jobs=args['n_jobs'])(delayed(train_model_wrapper)(dataset, entities, args) for entities in ENTITIES[d_i])

if __name__ == '__main__':
    main()    