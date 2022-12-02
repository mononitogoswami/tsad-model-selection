import pandas as pd
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Union

import sys
sys.path.append('/zfsauton2/home/mgoswami/tsad-model-selection/src/')

from model_selection.rank_aggregation import trimmed_kemeny, kemeny, borda, trimmed_borda, partial_borda, trimmed_partial_borda, _get_reliability
from metrics.ranking_metrics import rank_by_centrality, rank_by_synthetic_anomlies, rank_by_forecasting_metrics, rank_by_metrics
from model_selection.model_selection_utils import rank_models
from evaluation.utils import *
from evaluation.constants import MODEL_NAMES, QUANTITIES_SYNTHETIC_PREDICTIONS, QUANTITIES_PREDICTIONS, N_MODELS, N_SYNTHETIC_QUANTITIES, N_QUANTITIES

def _valid_prediction_inputs(ranking_obj, 
                             n_models=N_MODELS,
                             n_synthetic_quantities=N_SYNTHETIC_QUANTITIES, 
                             n_quantities=N_QUANTITIES,
                             verbose=True):
    VALID_MODELS = True
    VALID_QUANTITIES = True
    
    if (len(ranking_obj.predictions.keys()) != n_models) or\
         (len(ranking_obj.synthetic_predictions.keys()) != n_models):
         VALID_MODELS = False
         if verbose:
            print(f'Fewer than {n_models} models trained! Skipping...')

    model_names = list(ranking_obj.predictions.keys())
    for mn in model_names:
        if (len(ranking_obj.predictions[mn].keys()) != n_quantities) and\
            (len(ranking_obj.synthetic_predictions[mn].keys()) == n_synthetic_quantities):
            VALID_QUANTITIES = False
            if verbose:
                print(f'Fewer than {n_quantities} quantities or {n_synthetic_quantities} synthetic quantities models trained! Skipping...')

    VALID = VALID_MODELS and VALID_QUANTITIES
    
    return VALID

def pool_predictions_of_entities(entities,
                                 save_dir,
                                 dataset,
                                 ignore_timesteps=64, 
                                 quantities_synthetic_predictions=QUANTITIES_SYNTHETIC_PREDICTIONS,
                                 quantities_predictions=QUANTITIES_PREDICTIONS, 
                                 model_names=MODEL_NAMES):
    """Pool the synthetic predictions of multiple entities
    
    Parameters
    ----------
    ignore_timesteps: int
        Ignore the first few timesteps to allow models to warm up
    """

    predictions = {}
    synthetic_predictions = {}

    for mn in model_names:
        predictions[mn] = {}
        for q in quantities_predictions:
            predictions[mn][q] = []

        synthetic_predictions[mn] = {}
        for q in quantities_synthetic_predictions:
            synthetic_predictions[mn][q] = []

    for entity in tqdm(entities):
        ranking_obj_file = f'ranking_obj_{entity}.data'

        with open(os.path.join(save_dir, dataset, ranking_obj_file), 'rb') as f:
            ranking_obj = pkl.load(f)

        if not _valid_prediction_inputs(ranking_obj=ranking_obj, verbose=True):
            continue
        
        # NOTE: We are not normalizing the anomaly scores of the models
        # NOTE: Ignore the first few timesteps
        for mn in model_names:  
            for q in quantities_predictions:
                pred = ranking_obj.predictions[mn][q]        
                predictions[mn][q].append(pred[:, ignore_timesteps:])  
            
            for q in quantities_synthetic_predictions:
                pred = ranking_obj.synthetic_predictions[mn][q]
                if len(pred.shape) == 1:
                    pred = pred.reshape((1, -1))
                synthetic_predictions[mn][q].append(pred[:, ignore_timesteps:])  

    for mn in model_names:
        for q in quantities_predictions:
            predictions[mn][q] = np.concatenate(predictions[mn][q], axis=1)
        for q in quantities_synthetic_predictions:
            synthetic_predictions[mn][q] = np.concatenate(synthetic_predictions[mn][q], axis=1)
    
    return predictions, synthetic_predictions


def evaluate_models_pooled(entities: List[str],
                           save_dir: str=r'/home/scratch/mgoswami/results/',
                           dataset: str='smd',
                           ignore_timesteps: int=64,                           
                           n_neighbors: Union[List[int], int]=[2, 4, 6],
                           n_splits=100, 
                           sliding_window:int=None) -> pd.DataFrame:
    """
    Evaluates models in terms of 
    """

    # Pool predictions and synthetic predictions of entities
    predictions, synthetic_predictions = pool_predictions_of_entities(
        entities=entities,
        save_dir=save_dir,
        dataset=dataset,
        ignore_timesteps=ignore_timesteps)

    # Now use to predictions to rank the model
    models_metrics = rank_by_metrics(predictions, n_splits=n_splits, sliding_window=sliding_window)
    models_forecasting_metrics = rank_by_forecasting_metrics(predictions)
    models_centrality = rank_by_centrality(predictions,
                                           n_neighbors=n_neighbors)
    models_synthetic_anomlies = rank_by_synthetic_anomlies(
        synthetic_predictions,
        n_splits=n_splits)

    models_performance_matrix = pd.concat([
        models_metrics, 
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
                                      n_splits, 
                                      evaluation_metric, 
                                      metric, 
                                      top_k, 
                                      top_kr, 
                                      sliding_window, 
                                      use_all_ranks):
    aggregate_stats = {}
    models_performance_matrix_select = evaluate_models_pooled(
        entities=select_entities,
        dataset=dataset,
        save_dir=save_dir,
        n_neighbors=n_neighbors,
        n_splits=n_splits,
        sliding_window=sliding_window)
    
    models_performance_matrix_eval = evaluate_models_pooled(
        entities=eval_entities,
        dataset=dataset,
        save_dir=save_dir,
        n_neighbors=n_neighbors,
        n_splits=n_splits,
        sliding_window=sliding_window)

    ranks_by_metrics, *_ = rank_models(
        models_performance_matrix_eval
    )  # Rank Models based on the evaluation set

    if evaluation_metric == 'Best F-1':
        ranks = np.concatenate(
            [ranks_by_metrics[:8, :], ranks_by_metrics[8::3, :]],
            axis=0).astype(int)
    elif evaluation_metric == 'PR-AUC':
        ranks = np.concatenate(
            [ranks_by_metrics[:8, :], ranks_by_metrics[9::3, :]],
            axis=0).astype(int)
    elif evaluation_metric == 'VUS':
        ranks = np.concatenate(
            [ranks_by_metrics[:8, :], ranks_by_metrics[10::3, :]],
            axis=0).astype(int)

    performance_values = models_performance_matrix_eval.loc[:,
                                                            evaluation_metric].to_numpy(
                                                            ).squeeze()

    # Choose oracle based on selection set
    best_model_on_select_split = models_performance_matrix_select.index[
        np.argmax(models_performance_matrix_select.loc[:, evaluation_metric])]
    # Evaluate it on the evaluation set
    aggregate_stats['Oracle No-MS'] = models_performance_matrix_eval.loc[
        best_model_on_select_split, evaluation_metric]

    # Random Model Selection
    aggregate_stats['Random MS'] = np.mean(performance_values)
    aggregate_stats['Oracle MS'] = np.max(performance_values)

    metric_names = get_metric_names(models_performance_matrix_eval.columns,
                                    evaluation_metric=evaluation_metric)

    assert len(metric_names) == ranks.shape[
        0], f"Number of ranks should be equal to the number of metric names, rank shape: {ranks.shape}, names: {metric_names}"

    # Single Metric-based Model Selection
    for i, mn in enumerate(metric_names):
        aggregate_stats[f'{mn} MS'] = performance_values[ranks[i, :]][0]

    if not use_all_ranks:
        filtered_idxs = [
            i for i, mn in enumerate(metric_names)
            if ((len(mn.split('_')) == 3) and (mn.split(
                '_')[2] in ['noise', 'scale', 'cutoff', 'contextual', 'average']))
        ]
        ranks = ranks[filtered_idxs, :]

    # Rank-aggregation based Model Selection
    trimmed_kemeny_rank, kemeny_rank, trimmed_borda_rank, borda_rank,\
        partial_borda_rank, partial_trimmed_borda_rank, partial_trimmed_partial_borda_rank,\
             top_reliability_metric_rank, top_partial_reliability_metric_rank,\
                 reliability, partial_reliability = get_aggregated_ranks(
                    ranks=ranks, metric=metric, top_k=top_k, top_kr=top_kr)

    aggregate_stats['Trimmed Kemeny MS'] = performance_values[
        trimmed_kemeny_rank][0]
    aggregate_stats['Kemeny MS'] = performance_values[kemeny_rank][0]
    aggregate_stats['Trimmed Borda MS'] = performance_values[
        trimmed_borda_rank][0]
    aggregate_stats['Borda MS'] = performance_values[borda_rank][0]
    aggregate_stats['Partial Borda MS'] = performance_values[
        partial_borda_rank][0]
    aggregate_stats['Partial Trimmed Borda MS'] = performance_values[
        partial_trimmed_borda_rank][0]
    aggregate_stats['Partial Trimmed Partial Borda MS'] = performance_values[
        partial_trimmed_partial_borda_rank][0]
    aggregate_stats['Most Reliable Metric MS'] = performance_values[
        top_reliability_metric_rank][0]
    aggregate_stats['Most Reliable Metric (Partial) MS'] = performance_values[
        top_partial_reliability_metric_rank][0]
    
    return aggregate_stats

def _get_pooled_reliability(entities,           
                           dataset,
                           save_dir, 
                           evaluation_metric,
                           n_neighbors,
                           n_splits,
                           top_k, 
                           sliding_window):
    models_performance_matrix = evaluate_models_pooled(
        entities=entities,
        dataset=dataset,
        save_dir=save_dir,
        n_neighbors=n_neighbors,
        n_splits=n_splits,
        sliding_window=sliding_window)

    ranks_by_metrics, *_ = rank_models(models_performance_matrix)

    if evaluation_metric == 'Best F-1':
        ranks = np.concatenate(
            [ranks_by_metrics[:8, :], ranks_by_metrics[8::3, :]],
            axis=0).astype(int)
    elif evaluation_metric == 'PR-AUC':
        ranks = np.concatenate(
            [ranks_by_metrics[:8, :], ranks_by_metrics[9::3, :]],
            axis=0).astype(int)
    elif evaluation_metric == 'VUS':
        ranks = np.concatenate(
            [ranks_by_metrics[:8, :], ranks_by_metrics[10::3, :]],
            axis=0).astype(int)

    metric_names = get_metric_names(models_performance_matrix.columns,
                                    evaluation_metric=evaluation_metric)

    partial_reliability = _get_reliability(ranks=ranks,
                                        metric='influence',
                                        aggregation_type='partial_borda',
                                        top_k=top_k,
                                        n_neighbors=None)
    
    results = {
        'models_performance_matrix': models_performance_matrix, 
        'reliability': dict(zip(metric_names, partial_reliability)),
    }

    return results

#######################################
# Helper Functions for Evaluation
#######################################

def get_metric_names(performance_matrix_columns, evaluation_metric='PR-AUC'):
    if evaluation_metric == 'Best F-1': 
        evaluation_metric = 'F1'
        
    metric_names = [i for i in performance_matrix_columns[3:] if ((evaluation_metric in i) or ('CENTRALITY' in i) or (len(i.split('_')) == 1))]
    return metric_names

def get_aggregated_ranks(ranks: np.ndarray,
                         metric: str = 'influence',
                         top_k: int = 3,
                         top_kr: Optional[int] = None):
    """Get all kinds of aggregated ranks

    Parameters 
    ----------
    ranks: np.ndarray

    metric: str

    top_k: int
        Number of top ranks to consider for rank aggregation
    """
    # Trimmed Kemeny Rank Aggregation
    _, trimmed_kemeny_rank = trimmed_kemeny(ranks,
                                            metric=metric,
                                            aggregation_type='kemeny',
                                            verbose=False,
                                            top_kr=top_kr)
    trimmed_kemeny_rank = trimmed_kemeny_rank.astype(int)

    # Kemeny Rank Aggregation
    _, kemeny_rank = kemeny(ranks, verbose=False)
    kemeny_rank = kemeny_rank.astype(int)

    # Trimmed Borda Rank Aggregation
    _, trimmed_borda_rank = trimmed_borda(ranks,
                                          metric=metric,
                                          aggregation_type='borda',
                                          top_kr=top_kr)
    trimmed_borda_rank = trimmed_borda_rank.astype(int)

    # Borda Rank Aggregation
    _, borda_rank = borda(ranks)
    borda_rank = borda_rank.astype(int)

    # Top-k Borda Rank Aggregation
    _, partial_borda_rank = partial_borda(ranks, top_k=top_k)
    partial_borda_rank = partial_borda_rank.astype(int)

    # Partial Trimmed Borda
    _, partial_trimmed_borda_rank = trimmed_partial_borda(
        ranks,
        top_k=top_k,
        metric='influence',
        aggregation_type='borda',
        top_kr=top_kr)
    partial_trimmed_borda_rank = partial_trimmed_borda_rank.astype(int)

    # Partial Trimmed Partial Borda
    _, partial_trimmed_partial_borda_rank = trimmed_partial_borda(
        ranks,
        top_k=top_k,
        metric='influence',
        aggregation_type='partial_borda',
        top_kr=top_kr)
    partial_trimmed_partial_borda_rank = partial_trimmed_partial_borda_rank.astype(
        int)

    # Highest reliability metric
    reliability = _get_reliability(ranks=ranks,
                                   metric='influence',
                                   aggregation_type='borda',
                                   top_k=top_k,
                                   n_neighbors=None)
    top_reliability_metric_rank = ranks[np.argmax(reliability), :]

    # Highest reliability metric with partial borda
    partial_reliability = _get_reliability(ranks=ranks,
                                           metric='influence',
                                           aggregation_type='partial_borda',
                                           top_k=top_k,
                                           n_neighbors=None)
    top_partial_reliability_metric_rank = ranks[
        np.argmax(partial_reliability), :]

    return trimmed_kemeny_rank, kemeny_rank, trimmed_borda_rank, borda_rank, partial_borda_rank, partial_trimmed_borda_rank, partial_trimmed_partial_borda_rank, top_reliability_metric_rank, top_partial_reliability_metric_rank, reliability, partial_reliability


