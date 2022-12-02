# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import os 
sys.path.append('/zfsauton2/home/mgoswami/PyMAD/') # TODO: Make this relative path maybe
sys.path.append('/zfsauton2/home/mgoswami/tsad-model-selection/src')
sys.path.append('/zfsauton2/home/mgoswami/tsad-model-selection/VUS/')

from model_selection.model_selection import RankModels
from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, ANOMALY_ARCHIVE_ENTITY_TO_DATA_FAMILY
from model_selection.utils import visualize_predictions, visualize_data
from model_selection.rank_aggregation import trimmed_borda, trimmed_kemeny, trimmed_partial_borda, borda, kemeny, partial_borda, influence, averagedistance
from metrics.metrics import evaluate_model_selection

import math
import numpy as np
import pandas as pd
from vus.models.feature import Window
from vus.metrics import get_range_vus_roc
from vus.utils.slidingWindows import find_length
from sklearn.preprocessing import MinMaxScaler

from metrics.ranking_metrics import rank_by_metrics 
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

### Load the ranking object
DATASET = 'anomaly_archive'
EXPERIMENT_DATE = 'Oct29'
SAVE_DIR = f'/home/scratch/mgoswami/Experiments_{EXPERIMENT_DATE}/results'

correlations = []
for entity in tqdm(ANOMALY_ARCHIVE_ENTITIES):
    ranking_obj_file = f'ranking_obj_{entity}.data'
    try:
        with open(os.path.join(SAVE_DIR, DATASET, ranking_obj_file), 'rb') as f: 
            rankingObj = pkl.load(f)
        
        slidingWindow = find_length(rankingObj.test_data.entities[0].Y.flatten())
        
        model_performance_matrix = rank_by_metrics(predictions=rankingObj.predictions)
        
        kendall_correlation, kendall_pvalue = kendalltau(model_performance_matrix.loc[:, 'Best F-1'], model_performance_matrix.loc[:, 'VUS'])
        spearman_correlation, spearman_pvalue = spearmanr(model_performance_matrix.loc[:, 'Best F-1'], model_performance_matrix.loc[:, 'VUS'])

        correlations.append({
            'entity': entity, 
            'model_performance_matrix': model_performance_matrix,
            'estimated_sliding_window': slidingWindow, 
            'kendall_correlation': kendall_correlation,
            'kendall_pvalue': kendall_pvalue,
            'spearman_correlation': spearman_correlation,
            'spearman_pvalue': spearman_pvalue,
        })

    except:
        correlations.append({
            'entity': entity, 
            'model_performance_matrix': None,
            'estimated_sliding_window': None, 
            'kendall_correlation': None,
            'kendall_pvalue': None,
            'spearman_correlation': None,
            'spearman_pvalue': None,
        })
    
    with open(os.path.join(SAVE_DIR, 'correlation.pkl'), 'wb') as f: 
        pkl.dump(correlations, f)

