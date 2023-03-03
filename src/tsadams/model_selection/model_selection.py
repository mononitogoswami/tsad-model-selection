#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from typing import Tuple, Union, List
import os
from tqdm import tqdm
import numpy as np
import torch as t

from ..datasets.load import load_data
from ..utils.model_selection_utils import evaluate_model, evaluate_model_synthetic_anomalies, get_eval_batchsizes, rank_models
from ..metrics.ranking_metrics import rank_by_centrality, rank_by_synthetic_anomlies, rank_by_forecasting_metrics, rank_by_metrics

class RankModels(object):
    # NOTE: trained_model_path, downsampling, min_length, root_dir and normalize
    # should be shared between Trainer and RankModels Class
    def __init__(
            self,
            dataset: str = 'anomaly_archive',
            entity: str = '128_UCR_Anomaly_GP711MarkerLFM5z2',
            trained_model_path: str = r'/home/scratch/mgoswami/trained_models/',
            downsampling: int = 10,
            min_length: int = 256,
            root_dir: str = r'/home/scratch/mgoswami/datasets/',
            normalize: bool = True,
            verbose: bool = False):

        self.verbose = verbose
        self.train_data = load_data(dataset=dataset,
                                    group='train',
                                    entities=[entity],
                                    downsampling=downsampling,
                                    min_length=min_length,
                                    root_dir=root_dir,
                                    normalize=normalize,
                                    verbose=verbose)
        self.test_data = load_data(dataset=dataset,
                                   group='test',
                                   entities=[entity],
                                   downsampling=downsampling,
                                   min_length=min_length,
                                   root_dir=root_dir,
                                   normalize=normalize,
                                   verbose=verbose)

        self.TRAINED_MODELS_PATH = os.path.join(trained_model_path, dataset,
                                                entity)
        print(f'Path to trained models: {self.TRAINED_MODELS_PATH}')

        self.MODEL_NAMES = [
            i.split('.')[0] for i in os.listdir(self.TRAINED_MODELS_PATH)
            if 'pth' in i.split('.')[1]
        ]
        print(f'Models in the model set: {self.MODEL_NAMES}')
        print(
            f'Total number of models in the model set: {len(self.MODEL_NAMES)}'
        )

        self.generated_syn_anomaly_params = False
        self.predictions = None
        self.synthetic_predictions = None
        self.models_performance_matrix = None

    def __str__(self):
        return 'I am a RankModels object!'

    def get_random_syn_anomaly_params(self):
        # Synthetic anomaly injection parameters
        if not self.generated_syn_anomaly_params:
            self.random_states = np.random.randint(0,
                                                   10000,
                                                   size=self.n_repeats)
            self.window_sizes = np.random.choice(a=[32, 64, 128],
                                                 replace=True,
                                                 size=self.n_repeats)
            self.generated_syn_anomaly_params = True

        if self.verbose:
            print('Synthetic Anomaly Injection Parameters:')
            print(
                f'Number of repeats: {self.n_repeats} | Random states: {self.random_states} | Window Sizes: {self.window_sizes}'
            )

        return self.random_states, self.window_sizes

    def evaluate_models(self,
                        n_neighbors: Union[List[int], int] = [2, 4, 6],
                        n_repeats: int = 3,
                        split: str = 'test',
                        synthetic_ranking_criterion: str = 'prauc',
                        n_splits: int = 100,
                        sliding_window: int = None) -> pd.DataFrame:
        self.n_repeats = n_repeats

        self.predictions = {}
        self.synthetic_predictions = {}
        _ = self.get_random_syn_anomaly_params()

        for model_name in tqdm(self.MODEL_NAMES):
            with open(
                    os.path.join(self.TRAINED_MODELS_PATH,
                                 f'{model_name}.pth'), 'rb') as f:
                model = t.load(f)
            model.eval()  # Set model in evaluation mode

            eval_batch_size = get_eval_batchsizes(model_name=model_name)
            # print(f'Model Name: {model_name} | Evaluation batch size: {eval_batch_size}')

            self.predictions[model_name] = evaluate_model(
                data=self.test_data if split == 'test' else self.train_data,
                model=model,
                model_name=model_name,
                padding_type='right',
                eval_batch_size=eval_batch_size)

            self.synthetic_predictions[
                model_name] = evaluate_model_synthetic_anomalies(
                    data=self.test_data
                    if split == 'test' else self.train_data,
                    model=model,
                    model_name=model_name,
                    padding_type='right',
                    eval_batch_size=eval_batch_size,
                    n_repeats=self.n_repeats,
                    random_states=self.random_states,
                    max_window_size=128,
                    min_window_size=8)

        # Now use to predictions to rank the model
        # self.models_prauc = rank_by_praucs(self.predictions)
        # self.models_f1 = rank_by_max_F1(self.predictions, n_splits=n_splits)
        # self.models_prauc_f1 = rank_by_prauc_f1(self.predictions, n_splits=n_splits)
        
        self.models_evaluation_metrics = rank_by_metrics(self.predictions,n_splits=n_splits, sliding_window=sliding_window)
        self.models_forecasting_metrics = rank_by_forecasting_metrics(
            self.predictions)
        self.models_centrality = rank_by_centrality(self.predictions,
                                                    n_neighbors=n_neighbors)
        self.models_synthetic_anomlies = rank_by_synthetic_anomlies(
            self.synthetic_predictions,
            criterion=synthetic_ranking_criterion,
            n_splits=n_splits)

        self.models_performance_matrix = pd.concat([
            self.models_evaluation_metrics, self.models_forecasting_metrics,
            self.models_centrality, self.models_synthetic_anomlies
        ],
                                                   axis=1)

        return self.models_performance_matrix

    def rank_models(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.models_performance_matrix is not None, "Call evaluate_models() to evaluate models first!"
        self.ranks_by_metrics, self.rank_prauc, self.rank_f1 = rank_models(
            self.models_performance_matrix)
        return self.ranks_by_metrics, self.rank_prauc, self.rank_f1
