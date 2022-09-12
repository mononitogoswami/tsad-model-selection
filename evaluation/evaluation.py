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

from distributions import mallows_kendall as mk
from model_selection.rank_aggregation import trimmed_kemeny, kemeny, borda, trimmed_borda, partial_borda, trimmed_partial_borda, _get_reliability
from metrics.ranking_metrics import rank_by_praucs, rank_by_centrality, rank_by_synthetic_anomlies, rank_by_forecasting_metrics, rank_by_max_F1, rank_by_prauc_f1
from model_selection.model_selection_utils import rank_models
from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES, MACHINES, ANOMALY_ARCHIVE_ENTITY_TO_DATA_FAMILY

import sys
sys.path.append('/home/ubuntu/PyMAD/')
from src.pymad.evaluation.numpy import adjusted_precision_recall_f1_auc


#######################################
# Pooled Evaluation
#######################################

MODEL_NAMES = ['RNN_1', 'RNN_2', 'RNN_3', 'RNN_4', 'LSTMVAE_1', 'LSTMVAE_2', 'LSTMVAE_3', 'LSTMVAE_4', 'NN_1', 'NN_2', 'NN_3', 'DGHL_1', 'DGHL_2', 'DGHL_3', 'DGHL_4', 'MD_1', 'RM_1', 'RM_2', 'RM_3']
ANOMALY_TYPES = ['average', 'contextual', 'cutoff', 'flip', 'noise', 'scale', 'speedup', 'wander']
QUANTITIES = ['anomalysizes_type_spikes', 'anomalylabels_type_spikes', 'entityscores_type_spikes', 'Ta_type_spikes', 'anomalysizes_type_contextual', 'anomalylabels_type_contextual', 'entityscores_type_contextual', 'Ta_type_contextual', 'anomalysizes_type_flip', 'anomalylabels_type_flip', 'entityscores_type_flip', 'Ta_type_flip', 'anomalysizes_type_speedup', 'anomalylabels_type_speedup', 'entityscores_type_speedup', 'Ta_type_speedup', 'anomalysizes_type_noise', 'anomalylabels_type_noise', 'entityscores_type_noise', 'Ta_type_noise', 'anomalysizes_type_cutoff', 'anomalylabels_type_cutoff', 'entityscores_type_cutoff', 'Ta_type_cutoff', 'anomalysizes_type_scale', 'anomalylabels_type_scale', 'entityscores_type_scale', 'Ta_type_scale', 'anomalysizes_type_wander', 'anomalylabels_type_wander', 'entityscores_type_wander', 'Ta_type_wander', 'anomalysizes_type_average', 'anomalylabels_type_average', 'entityscores_type_average', 'Ta_type_average']


def pool_random_repetitions(synthetic_predictions):
    """Function to pool multiple random repetitions and kinds of anomalies
    """
    pooled_synthetic_predictions = {}

    for mn in MODEL_NAMES:
        pooled_synthetic_predictions[mn] = {}
        for quantity in synthetic_predictions[mn].keys(): 
            anomaly_type = quantity.split('_')[2]
            measure = quantity.split('_')[0]

            if f'{measure}_type_{anomaly_type}' not in pooled_synthetic_predictions[mn].keys(): 
                pooled_synthetic_predictions[mn][f'{measure}_type_{anomaly_type}'] = []
            pooled_synthetic_predictions[mn][f'{measure}_type_{anomaly_type}'].append(synthetic_predictions[mn][quantity].reshape((1, -1))) 

        for k in pooled_synthetic_predictions[mn].keys():
            pooled_synthetic_predictions[mn][k] = np.concatenate(pooled_synthetic_predictions[mn][k], axis=1)
    
    return pooled_synthetic_predictions

def pool_predictions_of_entities(entities, quantities, save_dir, dataset, ignore_timesteps=64, model_names=MODEL_NAMES):
    """Pool the synthetic predictions of multiple entities
    
    Parameters
    ----------
    ignore_timesteps: int
        Ignore the first few timesteps to allow models to warm up
    
    quantities: list
        List of measurable quantities to pool
    
    """

    if 'Y' in quantities: type = 'predictions'
    else: type = 'synthetic_predictions'

    predictions = {}
    
    for mn in model_names: 
        predictions[mn] = {}
        for q in quantities: 
            predictions[mn][q] = []
    
    for entity in tqdm(entities):
        ranking_obj_file = f'ranking_obj_{entity}.data'

        with open(os.path.join(save_dir, dataset, ranking_obj_file), 'rb') as f: 
            ranking_obj = pkl.load(f)

        if len(ranking_obj.predictions.keys()) != len(model_names):
            print(f'Fewer than 19 models trained on {entity}! Skipping...')
            continue

        if len(ranking_obj.synthetic_predictions.keys()) != len(model_names):
            print(f'{len(ranking_obj.predictions.keys())} != {len(ranking_obj.synthetic_predictions.keys())}') 
            print(f'Fewer than 19 models trained on {entity}! Skipping...')
            continue

        if type == 'synthetic_predictions': # Pool all the repeititions of the synthetic anomaly injections
            synthetic_predictions = pool_random_repetitions(ranking_obj.synthetic_predictions)
        
        for mn in model_names: # NOTE: We are not normalizing the anomaly scores of the models
            for q in quantities:
                if type == 'predictions':
                    predictions[mn][q].append(ranking_obj.predictions[mn][q][:, ignore_timesteps:]) # NOTE: Ignore the first few timesteps
                elif type == 'synthetic_predictions':
                    pred = synthetic_predictions[mn][q]
                    if len(pred.shape) == 1: 
                        pred = pred.reshape((1, -1))
                    predictions[mn][q].append(pred[:, ignore_timesteps:]) # NOTE: Ignore the first few timesteps
                
    for mn in model_names: 
        for q in quantities: 
            predictions[mn][q] = np.concatenate(predictions[mn][q], axis=1)

    return predictions

# TESTING
# entities = ['machine-1-1', 'machine-1-2', 'machine-2-1', 'machine-2-2']
# QUANTITIES = ['entity_scores', 'Y', 'Y_hat', 'Y_sigma', 'anomaly_labels', 'mask']
# predictions = pool_predictions_of_entities(entities, QUANTITIES, save_dir, dataset, ignore_timesteps=64, model_names=MODEL_NAMES)

# with open(os.path.join(save_dir, dataset, f'ranking_obj_machine-1-1.data'), 'rb') as f: 
#         ranking_obj = pkl.load(f)
# QUANTITIES = list(ranking_obj.synthetic_predictions[MODEL_NAMES[0]].keys())

# synthetic_predictions = pool_predictions_of_entities(entities, QUANTITIES, save_dir, dataset, ignore_timesteps=64, model_names=MODEL_NAMES)

def evaluate_models_pooled(entities:List[str], 
                           save_dir:str=r'/home/ubuntu/efs/results/', 
                           dataset:str='smd', 
                           ignore_timesteps:int=64, 
                           model_names:List[str]=MODEL_NAMES, 
                           n_neighbors:Union[List[int], int]=[2, 4, 6], 
                           synthetic_ranking_criterion:str='prauc',
                           n_splits=100)->pd.DataFrame:
    # Pool predictions of entities
    quantities = ['entity_scores', 'Y', 'Y_hat', 'Y_sigma', 'anomaly_labels', 'mask']
    predictions = pool_predictions_of_entities(entities=entities, quantities=quantities, 
                        save_dir=save_dir, dataset=dataset, ignore_timesteps=ignore_timesteps, 
                        model_names=model_names)
    
    # Pool synthetic predictions of entities
    synthetic_predictions = pool_predictions_of_entities(entities=entities, quantities=QUANTITIES, 
                                save_dir=save_dir, dataset=dataset, ignore_timesteps=ignore_timesteps, 
                                model_names=model_names)
    
   # Now use to predictions to rank the model
    models_prauc_f1 = rank_by_prauc_f1(predictions, n_splits=n_splits)
    models_forecasting_metrics = rank_by_forecasting_metrics(predictions)
    models_centrality = rank_by_centrality(predictions, n_neighbors=n_neighbors)
    models_synthetic_anomlies = rank_by_synthetic_anomlies(synthetic_predictions, criterion=synthetic_ranking_criterion, n_splits=n_splits)

    models_performance_matrix = pd.concat([
        models_prauc_f1,
        models_forecasting_metrics, 
        models_centrality, 
        models_synthetic_anomlies], axis=1)
        
    return models_performance_matrix

# Testing
# entities = ['machine-1-1', 'machine-1-2', 'machine-2-1', 'machine-2-2']
# models_performance_matrix = evaluate_models_pooled(entities=entities, n_neighbors=[2, 4, 6], synthetic_ranking_criterion='f1', n_splits=100)

def _get_pooled_aggregate_stats_split(select_entities, 
                                      eval_entities, 
                                      dataset,
                                      save_dir,
                                      n_neighbors, 
                                      synthetic_ranking_criterion, 
                                      ignore_metric, 
                                      n_splits, 
                                      evaluation_metric, 
                                      metric, 
                                      top_k, 
                                      top_kr):
    aggregate_stats = {}                           
    models_performance_matrix_select = evaluate_models_pooled(entities=select_entities, dataset=dataset, save_dir=save_dir, 
                                            n_neighbors=n_neighbors, synthetic_ranking_criterion=synthetic_ranking_criterion, 
                                            n_splits=n_splits)
    models_performance_matrix_eval = evaluate_models_pooled(entities=eval_entities, dataset=dataset, save_dir=save_dir, 
                                            n_neighbors=n_neighbors, synthetic_ranking_criterion=synthetic_ranking_criterion, 
                                            n_splits=n_splits)
    
    ranks_by_metrics, *_ = rank_models(models_performance_matrix_eval) # Rank Models based on the evaluation set
    
    if evaluation_metric == 'Best F-1':
        ranks = np.concatenate([ranks_by_metrics[:8, :], ranks_by_metrics[8::2, :]], axis=0).astype(int)
    elif evaluation_metric == 'PR-AUC':
        ranks = np.concatenate([ranks_by_metrics[:8, :], ranks_by_metrics[9::2, :]], axis=0).astype(int) 
    
    performance_values = models_performance_matrix_eval.loc[:, evaluation_metric].to_numpy().squeeze()

    # Choose oracle based on selection set
    best_model_on_select_split = models_performance_matrix_select.index[np.argmax(models_performance_matrix_select.loc[:, evaluation_metric])]
    # Evaluate it on the evaluation set
    aggregate_stats['Oracle No-MS'] = models_performance_matrix_eval.loc[best_model_on_select_split, evaluation_metric]
    
    # Random Model Selection
    aggregate_stats['Random MS'] = np.mean(performance_values)
    aggregate_stats['Oracle MS'] = np.max(performance_values)
    
    metric_names = get_metric_names(models_performance_matrix_eval.columns, ignore_metric=ignore_metric)
    
    assert len(metric_names) == ranks.shape[0], "Number of ranks should be equal to the number of metric names"
    
    # Single Metric-based Model Selection
    for i, mn in enumerate(metric_names):
        aggregate_stats[f'{mn} MS'] = performance_values[ranks[i, :]][0]

    filtered_idxs = [i for i, mn in enumerate(metric_names) if ((len(mn.split('_'))==3) and (mn.split('_')[2] in ['noise', 'scale', 'cutoff', 'contextual', 'average']))]

    # DEBUGGING
    # print(np.array(metric_names)[filtered_idxs])
    ranks = ranks[filtered_idxs, :]
    # print(ranks.shape)

    # Rank-aggregation based Model Selection
    trimmed_kemeny_rank, kemeny_rank, trimmed_borda_rank,\
         borda_rank, partial_borda_rank, partial_trimmed_borda_rank,\
            partial_trimmed_partial_borda_rank, top_reliability_metric_rank,\
                top_partial_reliability_metric_rank = get_aggregated_ranks(ranks=ranks, metric=metric, top_k=top_k, top_kr=top_kr)

    aggregate_stats['Trimmed Kemeny MS'] = performance_values[trimmed_kemeny_rank][0]
    aggregate_stats['Kemeny MS'] = performance_values[kemeny_rank][0]
    aggregate_stats['Trimmed Borda MS'] = performance_values[trimmed_borda_rank][0]
    aggregate_stats['Borda MS'] = performance_values[borda_rank][0]
    aggregate_stats['Partial Borda MS'] = performance_values[partial_borda_rank][0]
    aggregate_stats['Partial Trimmed Borda MS'] = performance_values[partial_trimmed_borda_rank][0]
    aggregate_stats['Partial Trimmed Partial Borda MS'] = performance_values[partial_trimmed_partial_borda_rank][0]
    aggregate_stats['Most Reliable Metric MS'] = performance_values[top_reliability_metric_rank][0]
    aggregate_stats['Most Reliable Metric (Partial) MS'] = performance_values[top_partial_reliability_metric_rank][0]

    return aggregate_stats

def get_pooled_aggregate_stats(save_dir:str=r'/home/ubuntu/efs/results/', 
                               dataset:str='smd', 
                               data_family:str='', 
                               evaluation_metric:str='Best F-1', 
                               n_validation_splits:int=5, 
                               n_neighbors:Union[List[int], int]=[2, 4, 6],
                               random_state:int=13, 
                               n_splits:int=100, 
                               metric:str='influence', 
                               top_k:int=3,
                               top_kr:Optional[int]=None,
                               n_jobs:int=5):

    ranking_object_files = os.listdir(os.path.join(save_dir, dataset))
    evaluated_entities = ['_'.join(i.split('_')[2:]).split('.')[0] for i in ranking_object_files]
    if dataset == 'anomaly_archive':
        evaluated_entities = [i for i in evaluated_entities if ANOMALY_ARCHIVE_ENTITY_TO_DATA_FAMILY[i] == data_family]
    evaluated_entities_arr = np.array(evaluated_entities).reshape((-1, 1))

    if evaluation_metric == 'PR-AUC': 
        ignore_metric = 'Best F-1'
        synthetic_ranking_criterion = 'prauc'
    elif evaluation_metric == 'Best F-1': 
        ignore_metric = 'PR-AUC'
        synthetic_ranking_criterion = 'f1'

    kf = KFold(n_splits=n_validation_splits, random_state=random_state, shuffle=True)

    aggregate_stats_temp = Parallel(n_jobs=n_jobs)(delayed(_get_pooled_aggregate_stats_split)(
                                                        evaluated_entities_arr[select_index].reshape((-1,)), 
                                                        evaluated_entities_arr[eval_index].reshape((-1,)), 
                                                        dataset,
                                                        save_dir,
                                                        n_neighbors, 
                                                        synthetic_ranking_criterion,
                                                        ignore_metric, 
                                                        n_splits, 
                                                        evaluation_metric, 
                                                        metric, 
                                                        top_k, 
                                                        top_kr) for eval_index, select_index in tqdm(kf.split(evaluated_entities_arr)))

    # Now aggregate the results of all the folds
    aggregate_stats = {}
    metrics_names = list(aggregate_stats_temp[0].keys())
    for res in aggregate_stats_temp:
        for mn in metrics_names:
            if mn not in aggregate_stats.keys(): aggregate_stats[mn] = []
            aggregate_stats[mn].append(res[mn]) 
    
    return aggregate_stats

#######################################
# Helper Functions for Evaluation
#######################################

def get_metric_names(performance_matrix_columns, ignore_metric='PR-AUC'):
    if ignore_metric == 'Best F-1':
        ignore_metric = 'F1'
    metric_names = [i for i in performance_matrix_columns\
         if ((i!='PR-AUC') and (i!='Best F-1') and f'SYNTHETIC_{ignore_metric}' not in i)]
    return metric_names

def get_aggregated_ranks(ranks:np.ndarray, metric:str='influence', top_k:int=3, top_kr:Optional[int]=None):
    """Get all kinds of aggregated ranks

    Parameters 
    ----------
    ranks: np.ndarray

    metric: str

    top_k: int
        Number of top ranks to consider for rank aggregation
    """
    # Trimmed Kemeny Rank Aggregation
    _, trimmed_kemeny_rank = trimmed_kemeny(ranks, metric=metric, aggregation_type='kemeny', verbose=False, top_kr=top_kr)
    trimmed_kemeny_rank = trimmed_kemeny_rank.astype(int)
    
    # Kemeny Rank Aggregation
    _, kemeny_rank = kemeny(ranks, verbose=False)
    kemeny_rank = kemeny_rank.astype(int)
    
    # Trimmed Borda Rank Aggregation
    _, trimmed_borda_rank = trimmed_borda(ranks, metric=metric, aggregation_type='borda', top_kr=top_kr)
    trimmed_borda_rank = trimmed_borda_rank.astype(int)
    
    # Borda Rank Aggregation
    _, borda_rank = borda(ranks)
    borda_rank = borda_rank.astype(int)

    # Top-k Borda Rank Aggregation
    _, partial_borda_rank = partial_borda(ranks, top_k=top_k)
    partial_borda_rank = partial_borda_rank.astype(int)

    # Partial Trimmed Borda
    _, partial_trimmed_borda_rank = trimmed_partial_borda(ranks, top_k=top_k, metric='influence', aggregation_type='borda', top_kr=top_kr)
    partial_trimmed_borda_rank = partial_trimmed_borda_rank.astype(int)

    # Partial Trimmed Partial Borda
    _, partial_trimmed_partial_borda_rank = trimmed_partial_borda(ranks, top_k=top_k, metric='influence', aggregation_type='partial_borda', top_kr=top_kr)
    partial_trimmed_partial_borda_rank = partial_trimmed_partial_borda_rank.astype(int)

    # Highest reliability metric
    reliability = _get_reliability(ranks=ranks, metric='influence', aggregation_type='borda', top_k=top_k, n_neighbors=None)
    top_reliability_metric_rank = ranks[np.argmax(reliability), :]

    # Highest reliability metric with partial borda
    partial_reliability = _get_reliability(ranks=ranks, metric='influence', aggregation_type='partial_borda', top_k=top_k, n_neighbors=None)
    top_partial_reliability_metric_rank = ranks[np.argmax(partial_reliability), :]

    return trimmed_kemeny_rank, kemeny_rank, trimmed_borda_rank, borda_rank, partial_borda_rank, partial_trimmed_borda_rank, partial_trimmed_partial_borda_rank, top_reliability_metric_rank, top_partial_reliability_metric_rank

#######################################
# 
#######################################

def _get_stats_for_entity(entity, dataset, metric, evaluation_metric, save_dir, verbose, top_k, top_kr):
    if verbose: 
        print(42*"=")
        print(f'Evaluating entity: {entity}')
        print(42*"=")
    
    if evaluation_metric == 'PR-AUC': ignore_metric = 'Best F-1'
    elif evaluation_metric == 'Best F-1': ignore_metric = 'PR-AUC'

    ranking_obj_file = f'ranking_obj_{entity}.data'
        
    with open(os.path.join(save_dir, dataset, ranking_obj_file), 'rb') as f: 
        ranking_obj = pkl.load(f)
        
    ranks = ranking_obj.ranks_by_metrics.astype(int)
    
    if evaluation_metric == 'Best F-1':
        ranks = np.concatenate([ranks[:8, :], ranks[8::2, :]], axis=0) 
    elif evaluation_metric == 'PR-AUC':
        ranks = np.concatenate([ranks[:8, :], ranks[9::2, :]], axis=0) 
    
    rank_prauc = ranking_obj.rank_prauc.astype(int)
    rank_f1 = ranking_obj.rank_f1.astype(int)
    praucs = ranking_obj.models_performance_matrix.iloc[:, 0].to_numpy().squeeze()
    f1s = ranking_obj.models_performance_matrix.iloc[:, 1].to_numpy().squeeze()
    
    if ranks.shape[1] != 19: 
        print(f'Only {ranks.shape[1]} models trained on {entity}! Skipping...')
        return None
    
    # Get metric and model names
    metric_names = get_metric_names(ranking_obj.models_performance_matrix.columns, ignore_metric=ignore_metric)
    model_names = np.array(list(ranking_obj.models_performance_matrix.index))

    # Rank-aggregation based Model Selection
    trimmed_kemeny_rank, kemeny_rank, trimmed_borda_rank,\
         borda_rank, partial_borda_rank, partial_trimmed_borda_rank,\
            partial_trimmed_partial_borda_rank, top_reliability_metric_rank,\
                top_partial_reliability_metric_rank = get_aggregated_ranks(ranks=ranks, metric=metric, top_k=top_k, top_kr=top_kr)
            
    stats = {
        'Rank by PR-AUC': model_names[rank_prauc], 
        'Rank by Best F-1': model_names[rank_f1],
        'Trimmed Kemeny Rank': model_names[trimmed_kemeny_rank],
        'Trimmed Kemeny PR-AUC': praucs[trimmed_kemeny_rank],
        'Trimmed Kemeny Best F-1': f1s[trimmed_kemeny_rank],
        'Kemeny Rank': model_names[kemeny_rank],
        'Kemeny PR-AUC': praucs[kemeny_rank],
        'Kemeny Best F-1': f1s[kemeny_rank],
        'Borda Rank': model_names[borda_rank],
        'Borda PR-AUC': praucs[borda_rank],
        'Borda Best F-1': f1s[borda_rank],
        'Trimmed Borda Rank': model_names[trimmed_borda_rank],
        'Trimmed Borda PR-AUC': praucs[trimmed_borda_rank],
        'Trimmed Borda Best F-1': f1s[trimmed_borda_rank],
        'Partial Borda Rank': model_names[partial_borda_rank],
        'Partial Borda PR-AUC': praucs[partial_borda_rank],
        'Partial Borda Best F-1': f1s[partial_borda_rank],
        'Partial Trimmed Borda PR-AUC': praucs[partial_trimmed_borda_rank],
        'Partial Trimmed Borda Best F-1': f1s[partial_trimmed_borda_rank],
        'Partial Trimmed Partial Borda PR-AUC': praucs[partial_trimmed_partial_borda_rank],
        'Partial Trimmed Partial Borda Best F-1': f1s[partial_trimmed_partial_borda_rank],
        'Most Reliable Metric PR-AUC': praucs[top_reliability_metric_rank],
        'Most Reliable Metric F-1': f1s[top_reliability_metric_rank],
        'Most Reliable Metric (Partial) PR-AUC': praucs[top_partial_reliability_metric_rank],
        'Most Reliable Metric (Partial) F-1': f1s[top_partial_reliability_metric_rank],
        'Max PR-AUC': np.max(praucs),
        'Max Best F-1': np.max(f1s),
        'PR-AUC': praucs,
        'Best F-1': f1s,
    }

    for i, mn in enumerate(metric_names):
        stats[f'Predicted PR-AUC ({mn})'] = praucs[ranks[i, :]]
        stats[f'Predicted Best F-1 ({mn})'] = f1s[ranks[i, :]]
        stats[f'{mn} Rank'] = model_names[ranks[i, :]]

    return stats

def get_stats_dict(dataset:str='smd', 
                   metric:str='influence', 
                   evaluation_metric:str='Best F-1',
                   overwrite:bool=False, 
                   n_jobs:int=5, 
                   save_dir:str=r'/home/ubuntu/efs/results/',
                   verbose:bool=True, 
                   top_k:Optional[int]=3, 
                   top_kr:Optional[int]=None)->dict:
    """Function to get dictionary of model selection performance
    """
    ranking_object_files = os.listdir(os.path.join(save_dir, dataset))
    evaluated_entities = ['_'.join(i.split('_')[2:]).split('.')[0] for i in ranking_object_files]

    if verbose: 
        print(f'{len(evaluated_entities)} entities evaluated')
    
    PATH_TO_STATS_PKL = os.path.join(save_dir, f'stats_{dataset}_{metric}_{evaluation_metric}.pkl') if metric == 'influence' else os.path.join(save_dir, f'stats_{dataset}_{metric}_{evaluation_metric}.pkl')

    if not overwrite and os.path.exists(PATH_TO_STATS_PKL):
        with open(PATH_TO_STATS_PKL ,'rb') as f: 
            stats = pkl.load(f)
        return stats
    
    # Else let's create the stats dictionary
    stats = Parallel(n_jobs=n_jobs)(delayed(_get_stats_for_entity)(entity, dataset, metric, evaluation_metric, save_dir, verbose, top_k, top_kr) for entity in tqdm(evaluated_entities))
    stats = dict(zip(evaluated_entities, stats))

    with open(PATH_TO_STATS_PKL ,'wb') as f: 
        pkl.dump(stats, f)

    return stats

def get_aggregate_stats(stats:dict, evaluation_metric='Best F-1')->dict:
    """Computes aggregate statistics (per entity)
    """
    aggregate_stats = {'Trimmed Kemeny MS': [], 
                        'Kemeny MS': [], 
                        'Trimmed Borda MS': [], 
                        'Borda MS': [], 
                        'Partial Borda MS': [], 
                        'Oracle MS': [], 
                        'Random MS': [], 
                        'MSE-based MS': [], 
                        'Likelihood-based MS': [], 
                        'Centrality-based MS': [], 
                        'Contextual Anomaly Injection-based MS': [], 
                        'Spikes Anomaly Injection-based MS': [], 
                        'Oracle No-MS': []}
    # NOTE: We can include more singlular anomaly-kinds
    
    # Let's find the model which performs the best on average
    PERFORMANCE = []
    for entity in stats.keys():
        PERFORMANCE.append(stats[entity][evaluation_metric])

    PERFORMANCE = np.array(PERFORMANCE)
    best_model_idx = np.argmax(np.mean(PERFORMANCE, axis=0))
    
    e = list(stats.keys())[0]
    mn_contextual = [i for i in stats[e].keys() if (('contextual' in i) and (evaluation_metric in i))][0].split('(')[1].split(')')[0]
    mn_spikes = [i for i in stats[e].keys() if (('spikes' in i) and (evaluation_metric in i))][0].split('(')[1].split(')')[0]

    for entity in stats.keys():
        aggregate_stats['Trimmed Kemeny MS'].append(stats[entity][f'Trimmed Kemeny {evaluation_metric}'][0])
        aggregate_stats['Kemeny MS'].append(stats[entity][f'Kemeny {evaluation_metric}'][0]) 
        aggregate_stats['Trimmed Borda MS'].append(stats[entity][f'Trimmed Borda {evaluation_metric}'][0])
        aggregate_stats['Borda MS'].append(stats[entity][f'Borda {evaluation_metric}'][0]) 
        aggregate_stats['Partial Borda MS'].append(stats[entity][f'Partial Borda {evaluation_metric}'][0]) 
        aggregate_stats['Oracle MS'].append(stats[entity][f'Max {evaluation_metric}'])
        aggregate_stats['Random MS'].append(np.mean(stats[entity][f'{evaluation_metric}']))
        aggregate_stats['Oracle No-MS'].append(stats[entity][f'{evaluation_metric}'][best_model_idx])
    
        # Model selection based-on single metrics
        aggregate_stats['MSE-based MS'].append(stats[entity][f'Predicted {evaluation_metric} (MSE)'][0])
        aggregate_stats['Likelihood-based MS'].append(stats[entity][f'Predicted {evaluation_metric} (LIKELIHOOD)'][0])
        aggregate_stats['Centrality-based MS'].append(stats[entity][f'Predicted {evaluation_metric} (CENTRALITY_16)'][0])
        aggregate_stats['Contextual Anomaly Injection-based MS'].append(stats[entity][f'Predicted {evaluation_metric} ({mn_contextual})'][0])
        aggregate_stats['Spikes Anomaly Injection-based MS'].append(stats[entity][f'Predicted {evaluation_metric} ({mn_spikes})'][0])

    for k, v in aggregate_stats.items():
        aggregate_stats[k] = np.array(v)

    # Let's take ratios and clamp them
    keys = list(aggregate_stats.keys())
    for m in keys:
        if m == 'Oracle MS': continue
        aggregate_stats[f'{m}/Oracle MS'] = aggregate_stats[m]/(aggregate_stats['Oracle MS'] + 1e-6)
        aggregate_stats[f'{m}/Oracle MS'][aggregate_stats[f'{m}/Oracle MS'] > 1] = 0

    return aggregate_stats

def get_anomaly_scores_labels(save_dir, dataset):
    ranking_object_files = os.listdir(os.path.join(save_dir, dataset))
    evaluated_entities = ['_'.join(i.split('_')[2:]).split('.')[0] for i in ranking_object_files]

    anomaly_scores_all_entities = {}
    anomaly_labels_all_entities = {}

    for entity in tqdm(evaluated_entities):     
        
        ranking_obj_file = f'ranking_obj_{entity}.data'
        
        with open(os.path.join(save_dir, dataset, ranking_obj_file), 'rb') as f: 
            ranking_obj = pkl.load(f)

        model_names = list(ranking_obj.predictions.keys())
        if len(model_names) != 19: 
            print(f'Fewer than 19 models trained on {entity}! Skipping...')
            continue
        
        anomaly_scores_all_entities[entity] = {}
        anomaly_labels_all_entities[entity] = {}
        
        for mn in model_names:
            anomaly_scores = ranking_obj.predictions[mn]['entity_scores'].squeeze()
            anomaly_labels = ranking_obj.predictions[mn]['anomaly_labels'].squeeze()
            
            std_dev = np.std(anomaly_scores)
            anomaly_scores = anomaly_scores/std_dev # Normalize the entity scores

            anomaly_scores_all_entities[entity][mn] = anomaly_scores
            anomaly_labels_all_entities[entity][mn] = anomaly_labels

    # anomaly_scores_all_models = pd.DataFrame(anomaly_scores_all_entities).T.to_dict()
    # anomaly_labels_all_models = pd.DataFrame(anomaly_labels_all_entities).T.to_dict()

    return anomaly_scores_all_entities, anomaly_labels_all_entities #anomaly_scores_all_models, anomaly_labels_all_models


def evaluate_model_performance(entities, stats, anomaly_scores_all_entities, anomaly_labels_all_entities, rank_name=None, n_splits=100, best_model=None):
    # Perform Model Selection
    concat_scores = []
    concat_labels = []
    
    for entity in entities:
        _best_model = stats[entity][rank_name][0] if best_model is None else best_model
        scores = anomaly_scores_all_entities[entity][_best_model].squeeze()
        labels = anomaly_labels_all_entities[entity][_best_model].squeeze()
          
        concat_scores.append(scores)
        concat_labels.append(labels)
    
    concat_scores = np.concatenate(concat_scores)
    concat_labels = np.concatenate(concat_labels)

    _, _, F1, PR_AUC, _ = adjusted_precision_recall_f1_auc(y_true=concat_labels, y_scores=concat_scores, n_splits=n_splits)
    return F1, PR_AUC

def evaluate_all_model_performance(entities, anomaly_scores_all_entities, anomaly_labels_all_entities, return_value='mean', n_splits=100):
    # Select Random Model 
    concat_scores = []
    concat_labels = []
    
    for entity in entities:
        scores = np.stack(list(anomaly_scores_all_entities[entity].values()))
        labels = np.stack(list(anomaly_labels_all_entities[entity].values()))

        concat_scores.append(scores)
        concat_labels.append(labels)

    concat_scores = np.concatenate(concat_scores, axis=1)    
    concat_labels = np.concatenate(concat_labels, axis=1)    

    F1s = []
    PR_AUCs = []
    for i in range(concat_scores.shape[0]):
        _, _, F1, PR_AUC, _ = adjusted_precision_recall_f1_auc(y_true=concat_labels[i, :], y_scores=concat_scores[i, :], n_splits=n_splits)
        F1s.append(F1)
        PR_AUCs.append(PR_AUC)
   
    if return_value == 'mean':
        return np.mean(F1s), np.mean(PR_AUCs)
    elif return_value == 'argmax':
        return np.argmax(F1s), np.argmax(PR_AUCs)

def get_aggregate_stats_concat(stats, anomaly_scores_all_entities, anomaly_labels_all_entities, evaluation_metric='Best F-1', n_splits=5, random_state=0):
    aggregate_stats = {'Trimmed Kemeny MS': [], 
                        'Kemeny MS': [], 
                        'Trimmed Borda MS': [], 
                        'Borda MS': [], 
                        'Partial Borda MS': [], 
                        'Random MS': [], 
                        'MSE-based MS': [], 
                        'Likelihood-based MS': [], 
                        'Centrality-based MS': [], 
                        'Contextual Anomaly Injection-based MS': [], 
                        'Spikes Anomaly Injection-based MS': [], 
                        'Oracle MS': [], 
                        'Oracle No-MS': []
                        }

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    evaluated_entities_arr = np.array(list(stats.keys())).reshape((-1, 1))
    RANK_NAMES = ['Trimmed Kemeny', 'Kemeny', 'Trimmed Borda', 'Borda', 'Partial Borda']
    
    for eval_index, select_index in tqdm(kf.split(evaluated_entities_arr)):
        # Model Selection based on Rank Aggregation
        entities = evaluated_entities_arr[eval_index].reshape((-1,))
        for ranking in RANK_NAMES:
            rank_name = ranking + ' Rank'
            F1, PR_AUC = evaluate_model_performance(entities=entities, 
                            rank_name=rank_name, stats=stats, 
                            anomaly_scores_all_entities=anomaly_scores_all_entities, 
                            anomaly_labels_all_entities=anomaly_labels_all_entities, 
                            n_splits=100)
            if evaluation_metric == 'Best F-1': aggregate_stats[f'{ranking} MS'].append(F1)
            elif evaluation_metric == 'PR-AUC': aggregate_stats[f'{ranking} MS'].append(PR_AUC)


        # Model Selection based on Individual Metrics
        entities = evaluated_entities_arr[eval_index].reshape((-1,))
        F1, PR_AUC = evaluate_model_performance(entities=entities, rank_name='MSE Rank', stats=stats, 
                        anomaly_scores_all_entities=anomaly_scores_all_entities, 
                        anomaly_labels_all_entities=anomaly_labels_all_entities, 
                        n_splits=100)
        if evaluation_metric == 'Best F-1': aggregate_stats['MSE-based MS'].append(F1)
        elif evaluation_metric == 'PR-AUC': aggregate_stats['MSE-based MS'].append(PR_AUC)
        
        F1, PR_AUC = evaluate_model_performance(entities=entities, rank_name='LIKELIHOOD Rank', stats=stats, 
                        anomaly_scores_all_entities=anomaly_scores_all_entities, 
                        anomaly_labels_all_entities=anomaly_labels_all_entities, 
                        n_splits=100)
        if evaluation_metric == 'Best F-1': aggregate_stats['Likelihood-based MS'].append(F1)
        elif evaluation_metric == 'PR-AUC': aggregate_stats['Likelihood-based MS'].append(PR_AUC)

        F1, PR_AUC = evaluate_model_performance(entities=entities, rank_name='CENTRALITY_16 Rank', stats=stats, 
                        anomaly_scores_all_entities=anomaly_scores_all_entities, 
                        anomaly_labels_all_entities=anomaly_labels_all_entities, 
                        n_splits=100)
        if evaluation_metric == 'Best F-1': aggregate_stats['Centrality-based MS'].append(F1)
        elif evaluation_metric == 'PR-AUC': aggregate_stats['Centrality-based MS'].append(PR_AUC)
        
        e = list(stats.keys())[0]
        rank_name = [i for i in stats[e].keys() if (('Rank' in i) and ('contextual' in i))][0]
        F1, PR_AUC = evaluate_model_performance(entities=entities, rank_name=rank_name, stats=stats, 
                        anomaly_scores_all_entities=anomaly_scores_all_entities, 
                        anomaly_labels_all_entities=anomaly_labels_all_entities, 
                        n_splits=100)
        if evaluation_metric == 'Best F-1': aggregate_stats['Contextual Anomaly Injection-based MS'].append(F1)
        elif evaluation_metric == 'PR-AUC': aggregate_stats['Contextual Anomaly Injection-based MS'].append(PR_AUC)

        rank_name = [i for i in stats[e].keys() if (('Rank' in i) and ('spikes' in i))][0]
        F1, PR_AUC = evaluate_model_performance(entities=entities, rank_name=rank_name, stats=stats, 
                        anomaly_scores_all_entities=anomaly_scores_all_entities, 
                        anomaly_labels_all_entities=anomaly_labels_all_entities, 
                        n_splits=100)
        if evaluation_metric == 'Best F-1': aggregate_stats['Spikes Anomaly Injection-based MS'].append(F1)
        elif evaluation_metric == 'PR-AUC': aggregate_stats['Spikes Anomaly Injection-based MS'].append(PR_AUC)
        
        # Random Model Selection
        entities = evaluated_entities_arr[eval_index].reshape((-1,))
        F1, _ = evaluate_all_model_performance(entities=entities, 
                        anomaly_scores_all_entities=anomaly_scores_all_entities, 
                        anomaly_labels_all_entities=anomaly_labels_all_entities, 
                        return_value='mean', n_splits=100)
        aggregate_stats[f'Random MS'].append(F1)
        if evaluation_metric == 'Best F-1': aggregate_stats['Random MS'].append(F1)
        elif evaluation_metric == 'PR-AUC': aggregate_stats['Random MS'].append(PR_AUC)
        
        # Select best model from Selection Set
        entities = evaluated_entities_arr[select_index].reshape((-1,))
        best_model_idx, _ = evaluate_all_model_performance(entities=entities, 
                                anomaly_scores_all_entities=anomaly_scores_all_entities, 
                                anomaly_labels_all_entities=anomaly_labels_all_entities, 
                                return_value='argmax', n_splits=100)
        
        model_names = list(list(anomaly_scores_all_entities.values())[0].keys())
        best_model_name = model_names[best_model_idx]
        
        entities = evaluated_entities_arr[eval_index].reshape((-1,))
        F1, PR_AUC = evaluate_model_performance(entities=entities, stats=stats, 
                        anomaly_scores_all_entities=anomaly_scores_all_entities, 
                        anomaly_labels_all_entities=anomaly_labels_all_entities, 
                        n_splits=100, best_model=best_model_name, rank_name=None)
        aggregate_stats[f'Oracle No-MS'].append(F1)
        if evaluation_metric == 'Best F-1': aggregate_stats['Oracle No-MS'].append(F1)
        elif evaluation_metric == 'PR-AUC': aggregate_stats['Oracle No-MS'].append(PR_AUC)

    return aggregate_stats