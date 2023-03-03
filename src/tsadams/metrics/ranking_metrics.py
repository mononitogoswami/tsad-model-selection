#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Callable, List
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..distributions.mallows_kendall import distance as kendalltau_dist
from .metrics import mse, mae, smape, mape, prauc, gaussian_likelihood, get_range_vus_roc, best_f1_linspace, adjusted_precision_recall_f1_auc

from ..utils.vus_utils import find_length

######################################################
# Functions to compute ranks of models given their predictions
######################################################


def rank_by_centrality(predictions: dict,
                       n_neighbors: Union[List[int], int] = [2, 4, 6],
                       metric: Callable = kendalltau_dist) -> pd.DataFrame:
    """Rank models based on the centrality of their anomaly score (entity score) vectors. 
    
    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    
    n_neighbours:Union[List[int], int]=4
        Number of neighbours to use when computing centrality. By default use the 
        3 nearest neighbours. 
    
    metric:Callable=kendalltau_dist
        Distance metric. By default the Kendall's Tau distance is used.
    """
    CENTRALITY = {}
    MODEL_NAMES = list(predictions.keys())
    entity_score_matrix = np.stack(
        [predictions[mn]['entity_scores'].squeeze() for mn in MODEL_NAMES],
        axis=0)
    if isinstance(n_neighbors, int):
        n_neighbors = [n_neighbors]

    neigh = NearestNeighbors(n_neighbors=np.max(n_neighbors),
                             algorithm='ball_tree',
                             metric=metric)
    neigh.fit(entity_score_matrix)

    for nn in n_neighbors:
        CENTRALITY[f'CENTRALITY_{nn}'] = dict(
            zip(
                MODEL_NAMES,
                neigh.kneighbors(entity_score_matrix,
                                 n_neighbors=nn)[0].mean(axis=1)))

    return pd.DataFrame(CENTRALITY)


def rank_by_metrics(predictions: dict, n_splits=100, sliding_window=None) -> pd.DataFrame:
    """Rank models based on their observed (adjusted) PR-AUCs, Best F-1 and VUS. 
    
    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    """
    MODEL_NAMES = list(predictions.keys())
    METRICS = {}
    METRICS['PR-AUC'] = {}
    METRICS['Best F-1'] = {}
    METRICS['VUS'] = {}
    for model_name in MODEL_NAMES:
        labels = predictions[model_name]['anomaly_labels'].squeeze()
        scores = predictions[model_name]['entity_scores'].squeeze()
        _, _, f1, auc, *_ = adjusted_precision_recall_f1_auc(labels, scores, n_splits)
        
        if sliding_window is None: 
            sliding_window = find_length(predictions[model_name]['Y'].flatten()) 

        scores = MinMaxScaler(feature_range=(0,1)).fit_transform(scores.reshape(-1,1)).ravel()
        evaluation_scores = get_range_vus_roc(scores, labels, sliding_window)
       
        METRICS['PR-AUC'][model_name] = auc
        METRICS['Best F-1'][model_name] = f1
        METRICS['VUS'][model_name] = evaluation_scores['VUS_ROC']

    return pd.DataFrame(METRICS)


def rank_by_praucs(predictions: dict, n_splits=100) -> pd.DataFrame:
    """Rank models based on their observed PR-AUCs. 
    
    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    """
    MODEL_NAMES = list(predictions.keys())
    PR_AUCS = {}
    PR_AUCS['PR-AUC'] = {}
    for model_name in MODEL_NAMES:
        PR_AUCS['PR-AUC'][model_name] = prauc(
            Y=predictions[model_name]['anomaly_labels'].squeeze(),
            Y_scores=predictions[model_name]['entity_scores'].squeeze(),
            segment_adjust=True,
            n_splits=n_splits)

    return pd.DataFrame(PR_AUCS)


def rank_by_max_F1(predictions: dict, n_splits=100) -> pd.DataFrame:
    """Rank models based on their observed best F1. 
    
    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    
    """
    MODEL_NAMES = list(predictions.keys())
    F1 = {}
    F1['Best F-1'] = {}
    for model_name in MODEL_NAMES:
        f1, precision, recall, predict, _, best_threshold = best_f1_linspace(
            scores=predictions[model_name]['entity_scores'].squeeze(),
            labels=predictions[model_name]['anomaly_labels'].squeeze(),
            n_splits=n_splits,
            segment_adjust=True)
        F1['Best F-1'][model_name] = f1

    return pd.DataFrame(F1)

def rank_by_vus(predictions: dict, sliding_window: int=None) -> pd.DataFrame:
    """Rank models based on their volume under the ROC surface metric

    A recent study [1] claimed that VUS-ROC is the best metrtic in terms of separability, 
    consistency and robustness.

    NOTE: VUS-ROC for multivariate timeseries is untested. 
    
    Parameters
    ----------
    predictions: dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    sliding_window: int
        Sliding window parameter for the VUS metric
    
    References
    ----------
    [1] Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series 
        Anomaly Detection
    """
    MODEL_NAMES = list(predictions.keys())
    if sliding_window is None: 
        sliding_window = find_length(predictions[MODEL_NAMES[0]]['Y'].flatten()) 

    VUS = {}
    VUS['VUS'] = {}
    for model_name in MODEL_NAMES:
        scores = predictions[model_name]['entity_scores'].squeeze()
        labels = predictions[model_name]['anomaly_labels'].squeeze()
        scores = MinMaxScaler(feature_range=(0,1)).fit_transform(scores.reshape(-1,1)).ravel()
        evaluation_scores = get_range_vus_roc(scores, labels, sliding_window)
        
        VUS['VUS'][model_name] = evaluation_scores['VUS_ROC']

    return pd.DataFrame(VUS)


def rank_by_forecasting_metrics(predictions: dict) -> pd.DataFrame:
    """Rank models based on their forecasting performance. 
    
    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    
    """
    FORECASTING_METRICS = {}
    MODEL_NAMES = list(predictions.keys())

    for model_name in MODEL_NAMES:
        fm = {}
        fm['MAE'] = mae(Y=predictions[model_name]['Y'],
                        Y_hat=predictions[model_name]['Y_hat'],
                        Y_sigma=predictions[model_name]['Y_sigma'],
                        mask=predictions[model_name]['mask'])
        fm['MSE'] = mse(Y=predictions[model_name]['Y'],
                        Y_hat=predictions[model_name]['Y_hat'],
                        Y_sigma=predictions[model_name]['Y_sigma'],
                        mask=predictions[model_name]['mask'])
        fm['SMAPE'] = smape(Y=predictions[model_name]['Y'],
                            Y_hat=predictions[model_name]['Y_hat'],
                            Y_sigma=predictions[model_name]['Y_sigma'],
                            mask=predictions[model_name]['mask'])
        fm['MAPE'] = mape(Y=predictions[model_name]['Y'],
                          Y_hat=predictions[model_name]['Y_hat'],
                          Y_sigma=predictions[model_name]['Y_sigma'],
                          mask=predictions[model_name]['mask'])
        fm['LIKELIHOOD'] = gaussian_likelihood(
            Y=predictions[model_name]['Y'],
            Y_hat=predictions[model_name]['Y_hat'],
            Y_sigma=predictions[model_name]['Y_sigma'],
            mask=predictions[model_name]['mask'])

        FORECASTING_METRICS[model_name] = fm

    return pd.DataFrame(FORECASTING_METRICS).T


def rank_by_synthetic_anomlies(predictions,
                               n_splits=100, 
                               sliding_window=None) -> pd.DataFrame:
    MODEL_NAMES = list(predictions.keys())
    ANOMALY_TYPES = list(
        set([i.split('_')[2] for i in predictions[MODEL_NAMES[0]].keys()]))

    evaluation_scores = {}

    for model_name in MODEL_NAMES:
        es = {}
        for anomaly_type in ANOMALY_TYPES:
            labels = predictions[model_name][
                f'anomalylabels_type_{anomaly_type}'].flatten()
            scores = predictions[model_name][
                f'entityscores_type_{anomaly_type}'].flatten()
            
            _, _, f1, auc, *_ = adjusted_precision_recall_f1_auc(labels, scores, n_splits)

            if sliding_window is None: 
                T_a = predictions[model_name][f'Ta_type_{anomaly_type}'].flatten()
                sliding_window = find_length(T_a) 

            scores = MinMaxScaler(feature_range=(0,1)).fit_transform(scores.reshape(-1,1)).ravel()
            vus_scores = get_range_vus_roc(scores, labels, sliding_window)
            
            es[f'SYNTHETIC_F1_{anomaly_type}'] = f1
            es[f'SYNTHETIC_PR-AUC_{anomaly_type}'] = auc
            es[f'SYNTHETIC_VUS_{anomaly_type}'] = vus_scores['VUS_ROC']

        evaluation_scores[model_name] = es

    return pd.DataFrame(evaluation_scores).T.dropna(axis=1)
