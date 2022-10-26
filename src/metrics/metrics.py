#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

######################################################
# Metrics to evaluate the quality of model selection
######################################################

import numpy as np
from typing import Optional, List, Union
from sklearn.metrics import ndcg_score, precision_recall_curve, auc, average_precision_score
from scipy.stats import spearmanr, kendalltau, norm
# import sys
# sys.path.append('/home/ubuntu/PyMAD/')
from src.pymad.evaluation.numpy import adjusted_precision_recall_f1_auc


def kendalltau_topk(a: np.array, b: np.array, k: int = 60):
    """Kendall's Tau correlation between the top-k elements according to a
    """
    idxs = np.argsort(-a)[:k]
    return kendalltau(a[idxs], b[idxs])


def ndcg(y_true: np.ndarray,
         y_score: np.ndarray,
         top_k: Optional[int] = None) -> float:
    # Normalized Discounted Cumulative Gain of the predicted model scores
    return ndcg_score(y_true=y_true,
                      y_score=y_score,
                      ignore_ties=True,
                      k=top_k)


def average_prauc(praucs: Union[List, np.ndarray]):
    # Average PR-AUCs of all the models
    return np.mean(praucs)


def mean_reciprocal_rank(y_true: np.ndarray,
                         y_score: np.ndarray,
                         top_k: Optional[int] = None) -> float:
    """
    https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """
    if top_k is None: top_k = 1

    sorted_model_preferences = sorted(zip(np.arange(len(y_score)), y_score),
                                      key=lambda x: x[1],
                                      reverse=True)
    sorted_model_pred_ranks = [i[0] for i in sorted_model_preferences]
    sorted_model_praucs = sorted(zip(np.arange(len(y_true)), y_true),
                                 key=lambda x: x[1],
                                 reverse=True)
    sorted_model_true_ranks = [i[0] for i in sorted_model_praucs]

    top_k_reciprocal_rank = 0
    for i in range(top_k):
        for rank, model_id in enumerate(sorted_model_true_ranks):
            if model_id == sorted_model_pred_ranks[i]:
                top_k_reciprocal_rank = top_k_reciprocal_rank + (1 /
                                                                 (rank + 1))

    return top_k_reciprocal_rank / top_k


METRICS_NAMES = [
    'PR-AUC of Top-1 Predicted Model', 'PR-AUC of Top-k Predicted Model',
    'Average PR-AUC', 'Range of PR-AUC', 'PR-AUC of Best Model',
    'PR-AUC of Top-3 Best Models', "Kendall's Tau Corr."
]


def get_metric_names():
    return METRICS_NAMES


def evaluate_model_selection(prauc: np.ndarray,
                             y_pred: np.ndarray,
                             k: int = 5) -> dict:
    """Evaluation metrics for model selection. 
    
    Parameters
    ----------
    prauc: np.ndarray (N,)
        PR-AUC of each model.
    y_pred: np.ndarray (N,)
        Predicted rank of each model.
    k: int
        Computes top-k accuracy and ndcg i.e. whether the chosen model is among Top-k according to 
        true model performance. 
    Returns
    ----------
    metrics: dict
        Dictionary of evaluation metrics
    """
    chosen_model_prauc = prauc[
        y_pred[0]]  # PR-AUC of the model with the highest preference
    chosen_top_k_prauc = np.mean(prauc[y_pred[:k]])
    mean_prauc = average_prauc(prauc)  # Mean PR-AUC of all the models
    highest_prauc = np.max(prauc)  # Highest PR-AUC among all the models
    highest_top_k_prauc = np.mean(prauc[np.argsort(-1 * prauc)[:k]])
    range_prauc = np.max(prauc) - np.min(prauc)

    corr_k, _ = kendalltau(x=prauc, y=y_pred)  # Kendall's Tau correlation

    metrics_values = [
        chosen_model_prauc, chosen_top_k_prauc, mean_prauc, range_prauc,
        highest_prauc, highest_top_k_prauc, corr_k
    ]

    return dict(zip(METRICS_NAMES, metrics_values))


######################################################
# Metrics to perform model selection
######################################################


def gaussian_likelihood(Y: np.ndarray,
                        Y_hat: np.ndarray,
                        Y_sigma: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        tol: float = 1e-6) -> float:
    if np.sum(np.isnan(Y_sigma)) > 0:
        pred_std = np.std(Y - Y_hat, axis=1, keepdims=True) + tol
    else:
        pred_std = Y_sigma + tol

    likelihood = norm.pdf((Y - Y_hat) / pred_std, loc=0, scale=1)
    likelihood = mask * likelihood
    return np.sum(likelihood) / np.sum(mask)  # Average likelihood


def mse(Y: np.ndarray,
        Y_hat: np.ndarray,
        Y_sigma: np.ndarray,
        mask: Optional[np.ndarray] = None) -> float:
    r"""
    Parameters
    ----------
    Y: np.ndarray
        Target values
    Y_hat: np.ndarray
        Predicted values
    Y_sigma: np.ndarray
        Predicted standard deviation
    mask: np.ndarray
        An array of 0s and 1s where 1 indicates which elements were masked for prediction. 
    
    .. math::

        mse = mean((Y - \hat{Y})^2)
    """
    if mask is None:
        return np.mean(np.square((Y - Y_hat)))
    else:
        return (np.sum(np.square(mask * (Y - Y_hat)))) / np.sum(mask)


def mae(Y: np.ndarray,
        Y_hat: np.ndarray,
        Y_sigma: np.ndarray,
        mask: Optional[np.ndarray] = None) -> float:
    r"""    
    .. math::

        mse = mean(|Y - \hat{Y}|)
    """
    if mask is None:
        return np.mean(np.abs((Y - Y_hat)))
    else:
        return (np.sum(np.abs(mask * (Y - Y_hat)))) / np.sum(mask)


def mape(Y: np.ndarray,
         Y_hat: np.ndarray,
         Y_sigma: np.ndarray,
         mask: Optional[np.ndarray] = None,
         tol: float = 1e-6) -> float:
    r"""
    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))
    """
    # Add small tolerance to ensure that division by |Y| does blow up
    Y = Y + tol
    Y_hat = Y_hat + tol
    if mask is None:
        return np.mean(np.abs(Y - Y_hat) / np.abs(Y))
    else:
        return (np.sum(mask * (np.abs(Y - Y_hat) / np.abs(Y)))) / np.sum(mask)


def smape(Y: np.ndarray,
          Y_hat: np.ndarray,
          Y_sigma: np.ndarray,
          mask: Optional[np.ndarray] = None,
          tol: float = 1e-6) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))
    """
    # Add small tolerance to ensure that division by |Y| does blow up
    Y = Y + tol
    Y_hat = Y_hat + tol
    if mask is None:
        return 2 * np.mean(np.abs(Y - Y_hat) / (np.abs(Y) + np.abs(Y_hat)))
    else:
        return 2 * (np.sum(mask *
                           (np.abs(Y - Y_hat) /
                            (np.abs(Y) + np.abs(Y_hat))))) / np.sum(mask)


def prauc(Y: np.ndarray,
          Y_scores: np.ndarray,
          segment_adjust: bool = True,
          n_splits: int = 100) -> float:
    r"""
    Compute the (adjusted) area under the precision-recall curve.
    
    Parameters
    ----------
    Y: np.ndarray
        Target values
    Y_scores: np.ndarray
        Predicted scores
    segment_adjust: bool
        Whether to compute adjusted PR-AUC. 
    n_splits: int
        Number of threshold splits to compute PR-AUC. 
    """
    if not segment_adjust:
        PR_AUC = average_precision_score(y_true=Y, probas_pred=Y_scores)
    else:
        PR_AUC = adjusted_precision_recall_f1_auc(Y,
                                                  Y_scores,
                                                  n_splits=n_splits)[3]

    return PR_AUC
