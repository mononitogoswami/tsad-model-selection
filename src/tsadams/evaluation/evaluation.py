#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from typing import List, Optional, Union

from ..model_trainer.entities import ANOMALY_ARCHIVE_ENTITY_TO_DATA_FAMILY
from ..utils.eval_utils import _get_pooled_aggregate_stats_split, _get_pooled_reliability

#######################################
# Pooled Evaluation
#######################################

def get_pooled_aggregate_stats(
        save_dir: str = r'/home/scratch/mgoswami/results/',
        dataset: str = 'smd',
        data_family: str = '',
        evaluation_metric: str = 'Best F-1',
        n_validation_splits: int = 5,
        n_neighbors: Union[List[int], int] = [2, 4, 6],
        random_state: int = 13,
        n_splits: int = 100,
        sliding_window: int = None,
        metric: str = 'influence',
        top_k: int = 3,
        top_kr: Optional[int] = None,
        n_jobs: int = 5, 
        use_all_ranks: bool = False):

    ranking_object_files = os.listdir(os.path.join(save_dir, dataset))
    evaluated_entities = [
        '_'.join(i.split('_')[2:]).split('.')[0] for i in ranking_object_files
    ]
    if dataset == 'anomaly_archive':
        evaluated_entities = [
            i for i in evaluated_entities
            if ANOMALY_ARCHIVE_ENTITY_TO_DATA_FAMILY[i] == data_family
        ]
    evaluated_entities_arr = np.array(evaluated_entities).reshape((-1, 1))

    kf = KFold(n_splits=n_validation_splits,
               random_state=random_state,
               shuffle=True)

    aggregate_stats_temp = Parallel(n_jobs=n_jobs)(
        delayed(_get_pooled_aggregate_stats_split)
        (select_entities=evaluated_entities_arr[select_index].reshape((-1, )), 
        eval_entities=evaluated_entities_arr[eval_index].reshape((-1, )),
        dataset=dataset, 
        save_dir=save_dir, 
        n_neighbors=n_neighbors, 
        n_splits=n_splits, 
        evaluation_metric=evaluation_metric, 
        metric=metric, 
        top_k=top_k, 
        top_kr=top_kr,
        sliding_window=sliding_window, 
        use_all_ranks=use_all_ranks)
        for eval_index, select_index in tqdm(kf.split(evaluated_entities_arr)))

    # Now aggregate the results of all the folds
    aggregate_stats = {}
    metrics_names = list(aggregate_stats_temp[0].keys())
    print(aggregate_stats_temp[0])
    for res in aggregate_stats_temp:
        for mn in metrics_names:
            if mn not in aggregate_stats.keys(): aggregate_stats[mn] = []
            aggregate_stats[mn].append(res[mn])

    return aggregate_stats

def get_pooled_reliability(
        save_dir: str = r'/home/scratch/mgoswami/results/',
        dataset: str = 'smd',
        data_family: str = '',
        evaluation_metric: str = 'Best F-1',
        n_neighbors: Union[List[int], int] = [2, 4, 6],
        random_state: int = 13,
        n_splits: int = 100,
        sliding_window: int = None,
        metric: str = 'influence',
        top_k: int = 3,
        n_jobs: int = 5):

    ranking_object_files = os.listdir(os.path.join(save_dir, dataset))
    evaluated_entities = [
        '_'.join(i.split('_')[2:]).split('.')[0] for i in ranking_object_files
    ]
    if dataset == 'anomaly_archive':
        evaluated_entities = [
            i for i in evaluated_entities
            if ANOMALY_ARCHIVE_ENTITY_TO_DATA_FAMILY[i] == data_family
        ]
    evaluated_entities_arr = np.array(evaluated_entities).reshape((-1, 1))
    aggregate_stats = _get_pooled_reliability(entities=evaluated_entities_arr.squeeze(), 
                                            dataset=dataset, 
                                            save_dir=save_dir, 
                                            evaluation_metric=evaluation_metric, 
                                            n_neighbors=n_neighbors, 
                                            n_splits=n_splits, 
                                            top_k=top_k, 
                                            sliding_window=sliding_window)

    return aggregate_stats