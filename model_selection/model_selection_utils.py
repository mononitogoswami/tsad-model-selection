#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List
import numpy as np
import torch as t
from tqdm import tqdm, trange
from copy import deepcopy
import sys
import pandas as pd
sys.path.append('/home/ubuntu/PyMAD/') # TODO: Make this relative path maybe
sys.path.append('/home/ubuntu/TSADModelSelection')

from src.pymad.loaders.loader import Loader
from src.pymad.datasets.dataset import Dataset, Entity
from src.pymad.models.base_model import PyMADModel

from model_selection.inject_anomalies import InjectAnomalies
from model_selection.utils import de_unfold
from sklearn.model_selection import ParameterGrid

######################################################
# Functions to predict Y_hat given a model
######################################################

def predict(batch:dict, model_name:str, model:PyMADModel)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a model and an input batch (Y), use the model to predict Y_hat.

    Parameters
    ----------
    batch: dict
        Input batch 
    model_name: str
        Name of the model to use for prediction
    """
    _VALID_MODEL_NAMES = ['RNN', 'DGHL', 'LSTMVAE', 'MD', 'RM', 'NN'] # TODO: Should be stored somewhere centrally
    
    model_type = model_name.split('_')[0]

    if model_type == 'RNN': 
        return _predict_rnn(batch, model)
    elif model_type == 'DGHL': 
        return _predict_dghl(batch, model)
    elif model_type == 'NN': 
        return _predict_nn(batch, model)
    elif model_type == 'MD': 
        return _predict_md(batch, model)
    elif model_type == 'RM': 
        return _predict_rm(batch, model)
    elif model_type == 'LSTMVAE': 
        return _predict_lstmvae(batch, model)
    else: 
        raise AttributeError(f'Model type must be one of {_VALID_MODEL_NAMES}, but {model_type} was passed!')

######################################################
# Functions to compute observations necessary to compute
# ranking metrics
######################################################

def evaluate_model(data:Union[Dataset, Entity], 
                   model:PyMADModel, 
                   model_name:str,
                   padding_type:str='right', 
                   eval_batch_size:int=128)->dict:
    """Compute observations necessary to evaluate a model on a given dataset.

    Description
    -----------
    This function computes predicted anomaly scores of an entity (entity scores), 
    Y_hat (the predicted values of the entity) and Y_sigma. Y_sigma is NaN in most
    cases except in the case of LSTM-VAE. These observations are useful for two
    classes of metrics, names forecasting error and cetrality. 

    Parameters
    ----------
    data:Union[Dataset, Entity]
        Dataset to evaluate the model on.   
    
    model:PyMADModel
        Model
    
    model_name:str
        Name of the model. 

    padding_type:str='right', 
        Padding type. By default, 'right'.
    
    eval_batch_size:int=32
        Evaluation batch size. By default, 32.

    Returns
    ---------
    PREDICTIONS: dict
        The prediction dictionary comprises of entity_scores, Y, Y_hat, Y_sigma, mask and anomaly_labels.
    """
    
    anomaly_labels = data.entities[0].labels
        
    dataloader = Loader(dataset=data, 
                        batch_size=eval_batch_size, 
                        window_size=model.window_size, 
                        window_step=model.window_step, 
                        shuffle=False, 
                        padding_type=padding_type,
                        sample_with_replace=False, 
                        verbose=False, 
                        mask_position='None', 
                        n_masked_timesteps=0) 
    
    if model.window_size == -1: window_size = data.entities[0].Y.shape[1]
    else: window_size = model.window_size
    
    entity_scores = t.zeros((len(dataloader), data.n_features, window_size))
    
    n_features = data.n_features
    if 'DGHL' in model_name and (data.entities[0].X is not None): # DGHL also considers covariates
        n_features = n_features + data.entities[0].X.shape[0]
    
    Y = np.zeros((len(dataloader), n_features, window_size))
    Y_hat = np.zeros((len(dataloader), n_features, window_size))
    Y_sigma = np.zeros((len(dataloader), n_features, window_size))
    mask = np.zeros((len(dataloader), n_features, window_size))
    
    step = 0
    for batch in dataloader:
        batch_size, n_features, window_size = batch['Y'].shape
        # Entity anomaly scores to compute PR-AUC and Centrality
        batch_anomaly_score = model.window_anomaly_score(input=batch, return_detail=True)
        entity_scores[ step:(step+batch_size), :, :] = batch_anomaly_score
        
        # Forecasting Error
        Y_b, Y_hat_b, Y_sigma_b, mask_b = predict(batch, model_name, model)
        Y[ step:(step+batch_size), :, :] = Y_b
        Y_hat[ step:(step+batch_size), :, :] = Y_hat_b
        Y_sigma[ step:(step+batch_size), :, :] = Y_sigma_b
        mask[ step:(step+batch_size), :, :] = mask_b
    
        step += batch_size

    # Final Anomaly Scores and forecasts
    entity_scores = model.final_anomaly_score(input=entity_scores, return_detail=False) # return_detail = False averages the anomaly scores across features. 
    entity_scores = entity_scores.detach().cpu().numpy()

    Y_hat = de_unfold(windows=Y_hat, window_step=model.window_step)
    Y = de_unfold(windows=Y, window_step=model.window_step)
    Y_sigma = de_unfold(windows=Y_sigma, window_step=model.window_step)
    mask = de_unfold(windows=mask, window_step=model.window_step)
    
    # Remove extra padding from Anomaly Scores and forecasts
    entity_scores = _adjust_scores_with_padding(scores=entity_scores, padding_size=dataloader.padding_size, padding_type=padding_type)

    Y_hat = _adjust_scores_with_padding(scores=Y_hat, padding_size=dataloader.padding_size, padding_type=padding_type)
    Y = _adjust_scores_with_padding(scores=Y, padding_size=dataloader.padding_size, padding_type=padding_type)
    Y_sigma = _adjust_scores_with_padding(scores=Y_sigma, padding_size=dataloader.padding_size, padding_type=padding_type)
    mask = _adjust_scores_with_padding(scores=mask, padding_size=dataloader.padding_size, padding_type=padding_type)

    return {'entity_scores': entity_scores, 'Y': Y, 'Y_hat': Y_hat, 'Y_sigma': Y_sigma, 'mask': mask, 'anomaly_labels': anomaly_labels}

def evaluate_model_synthetic_anomalies(data:Union[Dataset, Entity], 
                                       model:PyMADModel,
                                       model_name:str,
                                       padding_type:str='right', 
                                       eval_batch_size:int=128,
                                       n_repeats:int=3,
                                       random_states:List[int]=[0, 1, 2],
                                       max_window_size:int=128,
                                       min_window_size:int=8)->dict:
    """Compute observations necessary to evaluate a model on a given dataset with synthetic anomalies injected.

    Description
    -----------
    This function injects synthetic anomalies to the data and computes 
    computes predicted anomaly scores (entity scores). These observations 
    are useful to evaluate performance of a model on synthetic anomalies. 

    Parameters
    ----------
    data:Union[Dataset, Entity]
        Dataset to evaluate the model on.   
    
    model:PyMADModel
        Model
    
    model_name:str
        Name of the model. 

    padding_type:str='right'
        Padding type. By default, 'right'.
    
    eval_batch_size:int=32
        Evaluation batch size. By default, 32.

    n_repeats:int=3
        Number of indepent anomaly injection trials of each anomaly type. By default n_repeats=3. 
    
    random_states:List[int]=[0, 1, 2]
        Random seed for each trial. Constrols the anomaly injection. 
    
    max_window_size: int
        Maximum window size of injected anomaly
    
    min_window_size: int
        Miniumum window size of injected anomaly
    
    Returns
    ---------
    PREDICTIONS: dict
        The prediction dictionary comprises of entity_scores, Y (anomalous Y) and 
        anomalous scores returned by the anomaly injection algorithm. 
    """
    from model_selection.anomaly_parameters import ANOMALY_PARAM_GRID

    original_data = deepcopy(data)
    PREDICTIONS = {} 
    
    for i in trange(n_repeats): 
        for j, anomaly_params in enumerate(list(ParameterGrid(list(ANOMALY_PARAM_GRID.values())))):
            anomaly_obj = InjectAnomalies(random_state=random_states[i], 
                                          verbose=False, 
                                          max_window_size=max_window_size, 
                                          min_window_size=min_window_size)
            data = deepcopy(original_data)
            T = data.entities[0].Y        
            
            data_std = max(np.std(T), 0.01)

            anomaly_params['T'] = T
            anomaly_params['scale'] = anomaly_params['scale']*data_std
            anomaly_type = anomaly_params['anomaly_type']
            
            # Inject synthetic anomalies to the data
            T_a, anomaly_sizes, anomaly_labels =\
                anomaly_obj.inject_anomalies(**anomaly_params)
            
            anomaly_sizes = anomaly_sizes/data_std
            data.entities[0].Y = T_a
            data.entities[0].n_time = T_a.shape[1]
            data.entities[0].mask = np.ones((T_a.shape))
            data.total_time = T_a.shape[1]

            if model.window_size == -1: window_size = data.entities[0].Y.shape[1]
            else: window_size = model.window_size
    
            dataloader = Loader(dataset=data, 
                                batch_size=eval_batch_size, 
                                window_size=model.window_size, 
                                window_step=model.window_step, 
                                shuffle=False, 
                                padding_type=padding_type,
                                sample_with_replace=False, 
                                verbose=True, 
                                mask_position='None', 
                                n_masked_timesteps=0) 
        
            entity_scores = t.zeros((len(dataloader), data.n_features, window_size))

            step = 0
            for batch in dataloader:
                batch_size, _, _ = batch['Y'].shape
                batch_anomaly_score = model.window_anomaly_score(input=batch, return_detail=True)
                entity_scores[ step:(step+batch_size), :, :] = batch_anomaly_score
                step += batch_size

            # Final Anomaly Scores
            entity_scores = model.final_anomaly_score(input=entity_scores, return_detail=False) # return_detail = False averages the anomaly scores across features. 
            entity_scores = entity_scores.detach().cpu().numpy()
            
            # Remove extra padding from Anomaly Scores
            entity_scores = _adjust_scores_with_padding(scores=entity_scores, padding_size=dataloader.padding_size, padding_type=padding_type)
            
            PREDICTIONS[f'anomalysizes_type_{anomaly_type}_rep_{i}_{j}'] = anomaly_sizes
            PREDICTIONS[f'anomalylabels_type_{anomaly_type}_rep_{i}_{j}'] = anomaly_labels
            PREDICTIONS[f'entityscores_type_{anomaly_type}_rep_{i}_{j}'] = entity_scores
            PREDICTIONS[f'Ta_type_{anomaly_type}_rep_{i}_{j}'] = T_a

    return PREDICTIONS

######################################################
# Helper functions 
######################################################

def rank_models(models_performance_matrix:pd.DataFrame)->Tuple[np.ndarray, np.ndarray]:
    # If the value is lower for a model, the model is better
    LOWER_BETTER = ['MAE', 'MSE', 'SMAPE', 'MAPE', 'CENTRALITY'] 
    # If the value is higher for a model, the model is better
    HIGHER_BETTER = ['LIKELIHOOD', 'SYNTHETIC', 'PR-AUC', 'Best F-1']

    METRIC_NAMES = [i.split('_')[0] for i in models_performance_matrix.columns]
    SORT_DIRECTION = []
    for mn in METRIC_NAMES:
        if mn in HIGHER_BETTER: SORT_DIRECTION.append('Desc')
        elif mn in LOWER_BETTER: SORT_DIRECTION.append('Asc')
        else: raise ValueError('Undefined metric sort direction.')
        
    ranks = np.zeros(models_performance_matrix.shape).T

    for i, metric_name in enumerate(models_performance_matrix.columns):
        if SORT_DIRECTION[i] == 'Asc':
            ranks[i, :] = np.argsort(models_performance_matrix.loc[:, metric_name].to_numpy())
        elif SORT_DIRECTION[i] == 'Desc':
            ranks[i, :] = np.argsort(-models_performance_matrix.loc[:, metric_name].to_numpy())
    
    rank_prauc = ranks[0, :] # Rank based on PR-AUC
    rank_f1 = ranks[1, :] # Rank based on F-1
    ranks_by_metrics = ranks[2:, ] # Ranks includes 
    
    return ranks_by_metrics, rank_prauc, rank_f1

def get_eval_batchsizes(model_name:str)->int:
    """Return evaluation batch sizes of models
    """
    from model_trainer.hyperparameter_grids import RNN_TRAIN_PARAM_GRID, DGHL_TRAIN_PARAM_GRID, MD_TRAIN_PARAM_GRID, NN_TRAIN_PARAM_GRID, RM_TRAIN_PARAM_GRID, LSTMVAE_TRAIN_PARAM_GRID
    
    _VALID_MODEL_NAMES = ['RNN', 'DGHL', 'LSTMVAE', 'MD', 'RM', 'NN'] # TODO: Should be stored somewhere centrally
    
    model_type = model_name.split('_')[0]

    if model_type == 'RNN': 
        return RNN_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'DGHL': 
        return DGHL_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'NN': 
        return NN_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'MD': 
        return MD_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'RM': 
        return RM_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'LSTMVAE': 
        return LSTMVAE_TRAIN_PARAM_GRID['eval_batch_size'][0]
    else: 
        raise AttributeError(f'Model type must be one of {_VALID_MODEL_NAMES}, but {model_type} was passed!')

def _adjust_scores_with_padding(scores:np.ndarray, padding_size:int=0, padding_type:str='right'):
    if scores.ndim == 1: scores = scores[None, :]
    
    if (padding_type == 'right') and (padding_size > 0): 
        scores = scores[:, :-padding_size] 
    elif (padding_type == 'left') and (padding_size > 0): 
        scores = scores[:, padding_size:]
    return scores

######################################################
# Helper prediction functions for predict
######################################################

def _predict_base(batch, model):
    Y, Y_hat, mask = model.forward(batch)
    if isinstance(Y, t.Tensor): Y = Y.detach().cpu().numpy()
    if isinstance(Y_hat, t.Tensor): Y_hat = Y_hat.detach().cpu().numpy()
    if isinstance(mask, t.Tensor): mask = mask.detach().cpu().numpy()
    Y_sigma = np.NaN*np.ones(batch['Y'].shape)
    return Y, Y_hat, Y_sigma, mask

def _predict_dghl(batch, model):
    return _predict_base(batch, model)

def _predict_md(batch, model):
    return _predict_base(batch, model)

def _predict_nn(batch, model):
    return _predict_base(batch, model)

def _predict_rm(batch, model):
    return _predict_base(batch, model)

def _predict_rnn(batch, model):
    batch_size, n_features, window_size = batch['Y'].shape
    Y, Y_hat, mask = model.forward(batch)
    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()
    Y = Y.reshape(n_features, -1)[:,:window_size] # to [n_features, n_time]
    Y_hat = Y_hat.reshape(n_features, -1)[:,:window_size]  # to [n_features, n_time]
    mask = mask.reshape(n_features, -1)[:,:window_size]  # to [n_features, n_time]
    
    # Add mask dimension
    Y = Y[None, :, :]
    Y_hat = Y_hat[None, :, :]
    mask = mask[None, :, :]
    
    Y_sigma = np.NaN*np.ones(batch['Y'].shape)
    return Y, Y_hat, Y_sigma, mask

def _predict_lstmvae(batch, model):
    Y, Y_mu, mask, Y_sigma, *_ = model.forward(batch)
    Y, Y_hat, mask, Y_sigma  = Y.detach().cpu().numpy(), Y_mu.detach().cpu().numpy(), mask.detach().cpu().numpy(), Y_sigma.detach().cpu().numpy()
    return Y, Y_hat, Y_sigma, mask



