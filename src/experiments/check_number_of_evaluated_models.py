#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

######################################################
# Function to check the number of evaluated entities
######################################################

import os
from argparse import ArgumentParser
from tsadams.utils.config import Config
from tsadams.model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES

DATASETS = ['anomaly_archive', 'smd']
ENTITIES = [ANOMALY_ARCHIVE_ENTITIES, MACHINES]

def main():
    parser = ArgumentParser(description='Config file')
    parser.add_argument('--config_file_path',
                        '-c', 
                        type=str, 
                        default='../../config.yaml',
                        help='path to config file')
    args = parser.parse_args()
    args = Config(config_file_path=args.config_file_path).parse()

    total_models = 0
    for d, dataset in enumerate(DATASETS):
        n_evaluated_models = 0
        if not os.path.exists(os.path.join(args['results_path'], dataset)):
            print(f'No models evaluated for dataset {dataset}')
        else:
            n_evaluated_models = int(
                len(os.listdir(os.path.join(args['results_path'], dataset))))
            print(f'Total entities evaluated in {dataset} = {n_evaluated_models}')

        total_models = total_models + n_evaluated_models
    print(f'Total number of entities evaluated = {total_models}')

if __name__ == '__main__':
    main()
