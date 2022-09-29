#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy.stats import kendalltau
from joblib import Parallel, delayed
import time

import sys

sys.path.append('/home/ubuntu/PyMAD/')  # TODO: Make this relative path maybe
from src.pymad.logger import Logger
# Kendall’s tau is a measure of the correspondence between two rankings.
# Values close to 1 indicate strong agreement, and values close to -1 indicate strong disagreement.

# TODO: Look at the way the Rank aggregation based on pairwise comparison paper created synthetic datasets

n_metrics = 15  # Number of metrics
n_models = 20  # Number of models
n_samples = 50000  # Number of candidate rankings
random_seed = 13
n_jobs = 8
save_data_path = '/home/ubuntu/datasets/'

start = time.time()

np.random.seed(random_seed)

# Randomly initialize the PR-AUCs of models
PR_AUCs = np.random.rand(n_models)
# print(f'PR-AUCs: {PR_AUCs}')
ranked_models_scores = sorted(zip(np.arange(n_models), PR_AUCs),
                              key=lambda x: x[1],
                              reverse=True)
ranked_models_gt = np.array([i[0] for i in ranked_models_scores])
ranked_PRAUCs = np.array([i[1] for i in ranked_models_scores])
# print(f'Ground truth model ranking: {ranked_models_gt}')

# Randomly permute model rankings
random_permutations = np.array(
    [np.random.permutation(n_models) for i in range(n_samples)])

# Compute kendalltau's correleation coefficient for each of the model rankings
# kendalltau_corr = np.array([kendalltau(x=ranked_models_gt, y=random_permutations[i, :])[0] for i in range(n_samples)]) # Sequential


def compute_kendalltau(x, y):
    return kendalltau(x, y)[0]


kendalltau_corr = np.array(
    Parallel(n_jobs=n_jobs)(delayed(compute_kendalltau)(
        x=ranked_models_gt, y=random_permutations[i, :])
                            for i in range(n_samples)))  # Parallel

# Choose the top-m positive correlated rankings (m = n_metrics)
sorted_corr = sorted(zip(np.arange(n_samples), kendalltau_corr),
                     key=lambda x: x[1],
                     reverse=True)
sample_corr_rankings = np.array([i[0] for i in sorted_corr])[:n_metrics]
ranked_sample_corrs = np.array([i[1] for i in sorted_corr])[:n_metrics]

data_obj = {
    'model_pr_aucs': ranked_PRAUCs,
    'ranked_models_gt': ranked_models_gt,
    'ranking_by_metric': random_permutations[sample_corr_rankings, :],
    'metric_kendall_taus': ranked_sample_corrs,
}
# print(ranked_models_gt)
# print(random_permutations[sample_corr_rankings, :])
print(f'Highest ranking correlation: {np.max(ranked_sample_corrs)}')
print(f'Lowest ranking correlation: {np.min(ranked_sample_corrs)}')
print(
    f'Shape of ranking_by_metric: {random_permutations[sample_corr_rankings, :].shape}'
)

# Declare a logger object to save the data
logging_obj = Logger(save_dir=save_data_path, overwrite=True, verbose=True)
# Save the data
logging_obj.save(obj=data_obj,
                 obj_name=f"synthetic_rank_data",
                 obj_meta={
                     'n_metrics': n_metrics,
                     'n_models': n_models,
                     'n_samples': n_samples,
                     'random_seed': random_seed
                 },
                 obj_class=['synthetic_data'])
print(f'Time elapsed: {time.time() - start}')
