#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to generate ranking objects for datasets
#######################################

from joblib import Parallel, delayed
from tsadams.model_selection.model_selection import RankModels
from tsadams.utils.logger import Logger
from tsadams.utils.utils import get_args_from_cmdline
from tsadams.model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES
from tsadams.utils.set_all_seeds import set_all_seeds

def set_eval_params(args):
    # Logger object to save the models
    logging_obj = Logger(save_dir=args['results_path'],
                         overwrite=args['overwrite'],
                         verbose=args['verbose'])

    ### Parameters for our Ranking Model
    rank_model_params = {
        'trained_model_path': args['trained_model_path'],
        'min_length': args['min_length'],
        'root_dir': args['dataset_path'],
        'normalize': args['normalize'],
        'verbose': args['verbose']
    }

    ### Parameters for our Evaluation Model
    evaluate_model_params = {
        'n_repeats': args['n_repeats'],
        'n_neighbors': args['n_neighbors'],
        'split': args['split'],
        'synthetic_ranking_criterion': args['synthetic_ranking_criterion'], 
        'n_splits': args['n_splits'],
    }
    return evaluate_model_params, logging_obj, rank_model_params


def evaluate_model_wrapper(dataset, entity, args):
    unevaluated_entities = []  # List of entities which remain unevaluated
    evaluate_model_params, logging_obj, rank_model_params = set_eval_params(args)

    print(42 * "=")
    print(f"Evaluating models on entity: {entity}")
    print(42 * "=")

    rank_model_params['dataset'] = dataset
    rank_model_params['entity'] = entity
    rank_model_params['downsampling'] = args['downsampling']
    logging_hierarchy = [dataset]

    if not args['downsampling']:
        if logging_obj.check_file_exists(obj_class=logging_hierarchy,
                                         obj_name=f'ranking_obj_{entity}'):
            print(f'Models on entity {entity} already evaluated!')
            return

    # Create a ranking object
    ranking_obj = RankModels(**rank_model_params)

    try: 
        _ = ranking_obj.evaluate_models(**evaluate_model_params)
    except:
        print(f'Error in evaluating models on entity: {entity}')

    # Save the ranking objection for later use
    logging_obj.save(obj=ranking_obj,
                     obj_name=f'ranking_obj_{entity}',
                     obj_meta=None,
                     obj_class=logging_hierarchy,
                     type='data')

def main():
    args = get_args_from_cmdline()
    
    set_all_seeds(args['random_seed']) # Reduce randomness
    
    # DATASETS = ['smd', 'anomaly_archive']
    # ENTITIES = [MACHINES, ANOMALY_ARCHIVE_ENTITIES]
    DATASETS = ['anomaly_archive']
    ENTITIES = [ANOMALY_ARCHIVE_ENTITIES[::-1]]

    for d_i, dataset in enumerate(DATASETS):
        _ = Parallel(n_jobs=args['n_jobs'])(
            delayed(evaluate_model_wrapper)(dataset, entities, args)
            for entities in ENTITIES[d_i])

if __name__ == '__main__':
    main()    