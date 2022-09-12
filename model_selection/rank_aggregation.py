#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##########################################
# Functions for rank aggregation
##########################################

from re import L
import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
import cvxpy as cp
from itertools import combinations, permutations
import distributions.mallows_kendall as mk
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

##########################################
# Trimmed Rank Aggregators
##########################################

def trimmed_partial_borda(ranks:np.ndarray, 
                          weights:Optional[np.ndarray]=None, 
                          top_k:Optional[int]=None, 
                          top_kr:Optional[int]=None, 
                          aggregation_type='kemeny', 
                          metric:str='influence', 
                          n_neighbors:int=6)->Tuple[float, np.ndarray]:
    """Computes the trimmed borda rank

    Parameters
    ----------
    ranks: [# permutations, # items]
        Array of ranks
    
    weights: [# permutations,]
        Weights of each permutation. By default, weights=None.
    
    top_k: int
        How many items to consider for partial rank aggregation. 
        By default top_k=None. 

    top_kr: int
        How many permutations to use for rank aggregation. 
        By default top_kr=None. If top is None, then use 
        agglomerative clustering. 

    aggregation_type: int
        Type of aggregation method to use while computing influence.
        We recommend 'borda' in large problems and kemeny in smaller 
        problems.

    metric: str 
        Metric of rank reliablity. By default metric='influence'. 
    
    n_neighbors: int
        Number of neighbours to use for proximity based reliability
    """
    reliability = _get_reliability(ranks=ranks, metric=metric, 
                                   aggregation_type=aggregation_type, 
                                   top_k=top_k, n_neighbors=n_neighbors)

    if top_kr is None: 
        trimmed_ranks = _get_trimmed_ranks_clustering(ranks, reliability)
    else:
        trimmed_ranks = ranks[np.argsort(-1*reliability)[:top_kr], :]
    
    if weights is not None:
        trimmed_weights = weights[np.argsort(-1*reliability)[:top_kr], :]
    else:
        trimmed_weights = None
    
    return partial_borda(ranks=trimmed_ranks, weights=trimmed_weights, top_k=top_k)

def trimmed_borda(ranks:np.ndarray, 
                  weights:Optional[np.ndarray]=None, 
                  top_k:Optional[int]=None, 
                  top_kr:Optional[int]=None,  
                  aggregation_type='kemeny', 
                  metric:str='influence', 
                  n_neighbors:int=6)->Tuple[float, np.ndarray]:
    """Computes the trimmed borda rank

    Parameters
    ----------
    ranks: [# permutations, # items]
        Array of ranks
    
    weights: [# permutations,]
        Weights of each permutation. By default, weights=None.
    
    top_k: int
        How many items to consider for partial rank aggregation. 
        By default top_k=None. 

    top_kr: int
        How many permutations to use for rank aggregation. 
        By default top_kr=None. If top is None, then use 
        agglomerative clustering. 

    aggregation_type: int
        Type of aggregation method to use while computing influence.
        We recommend 'borda' in large problems and kemeny in smaller 
        problems.

    metric: str 
        Metric of rank reliablity. By default metric='influence'.
    
    n_neighbors: int
        Number of neighbours to use for proximity based reliability
    """
    reliability = _get_reliability(ranks=ranks, metric=metric, 
                                   aggregation_type=aggregation_type, 
                                   top_k=top_k, n_neighbors=n_neighbors)

    if top_kr is None: 
        trimmed_ranks = _get_trimmed_ranks_clustering(ranks, reliability)
    else:
        trimmed_ranks = ranks[np.argsort(-1*reliability)[:top_kr], :]
    
    if weights is not None:
        trimmed_weights = weights[np.argsort(-1*reliability)[:top_kr], :]
    else:
        trimmed_weights = None
    
    return borda(ranks=trimmed_ranks, weights=trimmed_weights)

def trimmed_kemeny(ranks:np.ndarray, 
                  weights:Optional[np.ndarray]=None, 
                  top_k:Optional[int]=None, 
                  top_kr:Optional[int]=None, 
                  aggregation_type='kemeny',
                  metric:str='influence', 
                  n_neighbors:int=6,
                  verbose:bool=True)->Tuple[float, np.ndarray]:
    """Computes the trimmed kemeny rank

    Parameters
    ----------
    ranks: [# permutations, # items]
        Array of ranks
    
    weights: [# permutations,]
        Weights of each permutation. By default, weights=None.
    
    top_k: int
        How many items to consider for partial rank aggregation. 
        By default top_k=None. 

    top_kr: int
        How many permutations to use for rank aggregation. 
        By default top_kr=None. If top is None, then use 
        agglomerative clustering. 

    aggregation_type: int
        Type of aggregation method to use while computing influence.
        We recommend 'borda' in large problems and kemeny in smaller 
        problems.

    metric: str 
        Metric of rank reliablity. By default metric='influence'.
    
    n_neighbors: int
        Number of neighbours to use for proximity based reliability

    verbose: bool
        Controls verbosity
    """
    reliability = _get_reliability(ranks=ranks, metric=metric, 
                                   aggregation_type=aggregation_type, 
                                   top_k=top_k, n_neighbors=n_neighbors)

    if top_kr is None: 
        trimmed_ranks = _get_trimmed_ranks_clustering(ranks, reliability)
    else:
        trimmed_ranks = ranks[np.argsort(-1*reliability)[:top_kr], :]
    
    if weights is not None:
        trimmed_weights = weights[np.argsort(-1*reliability)[:top_kr], :]
    else:
        trimmed_weights = None
    
    return kemeny(ranks=trimmed_ranks, weights=trimmed_weights, verbose=verbose)
    
##########################################
# Rank Aggregators
##########################################

def partial_borda(ranks:np.ndarray, 
                  weights:Optional[np.ndarray]=None, 
                  top_k:int=5)->Tuple[float, np.ndarray]:
    # Top-k Borda Rank Aggregation
    # NOTE: weights is only for compatibility, currently not using weights
    
    ranks = ranks.astype(float)
    ranks = np.nan_to_num(x=ranks, nan=ranks.shape[1]+1) # If ranks already have NaNs
    # Mask higher ranks
    x, y = np.where((ranks > (top_k-1))) 
    for x_i, y_i in zip(x, y): ranks[x_i, y_i] = np.NaN
    aggregated_rank = np.nan_to_num(x=mk.borda_partial(ranks, w=1, k=top_k), nan=ranks.shape[1]-1).astype(int)
    objective = np.mean([mk.distance(r, aggregated_rank) for r in ranks])
    
    return objective, aggregated_rank

def borda(ranks:np.ndarray, 
          weights:Optional[np.ndarray]=None)->Tuple[float, np.ndarray]:
    if weights is None: 
        aggregated_rank = mk.median(ranks)
    else: 
        aggregated_rank = mk.weighted_median(ranks)
    objective = np.mean([mk.distance(r, aggregated_rank) for r in ranks])
    return objective, aggregated_rank

def kemeny(ranks:np.ndarray, 
           weights:Optional[np.ndarray]=None, 
           verbose:bool=True)->Tuple[float, np.ndarray]:
    """Kemeny-Young optimal rank aggregation [1]

    We include the ability to incorporate weights of metrics/permutations. 

    Parameters
    ----------
    ranks: 
        Permutations/Ranks
    weights:
        Weight of each rank/permutation. 
    verbose:
        Controls verbosity

    References
    ----------
    [1] Conitzer, V., Davenport, A., & Kalagnanam, J. (2006, July). Improved bounds for computing Kemeny rankings. In AAAI (Vol. 6, pp. 620-626).
        https://www.aaai.org/Papers/AAAI/2006/AAAI06-099.pdf
    [2] http://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html#note4\
    """

    _, n_models = ranks.shape

    # Minimize C.T * X
    edge_weights = build_graph(ranks, weights)
    C = -1 * edge_weights.ravel().reshape((-1, 1))
    
    # Defining variables
    X = cp.Variable((n_models**2, 1), boolean=True)

    # Defining the objective function
    objective = cp.Maximize(C.T @ X)

    # Defining the constraints
    idx = lambda i, j: n_models * i + j

    # Constraints for every pair
    pairwise_constraints = np.zeros(((n_models * (n_models - 1)) // 2,
                                    n_models ** 2))
    for row, (i, j) in zip(pairwise_constraints, combinations(range(n_models), 2)):
        row[[idx(i, j), idx(j, i)]] = 1

    # and for every cycle of length 3
    triangle_constraints = np.zeros(((n_models * (n_models - 1) * (n_models - 2)), n_models ** 2))
    for row, (i, j, k) in zip(triangle_constraints, permutations(range(n_models), 3)):
        row[[idx(i, j), idx(j, k), idx(k, i)]] = 1

    constraints = []
    constraints += [pairwise_constraints @ X == np.ones((pairwise_constraints.shape[0], 1))]
    constraints += [triangle_constraints @ X >= np.ones((triangle_constraints.shape[0], 1))]
        
    # Solving the problem
    problem = cp.Problem(objective, constraints)
    
    if verbose: 
        print("Is DCP:", problem.is_dcp())
    problem.solve(verbose=verbose, warm_start=True)

    aggregated_rank = X.value.reshape((n_models, n_models)).sum(axis=1)

    objective = np.mean([mk.distance(r, aggregated_rank) for r in ranks])

    return objective, aggregated_rank

##########################################
# Helper functions
##########################################

def _get_reliability(ranks, metric='influence', aggregation_type='borda', top_k=None, n_neighbors=6):
    if metric == 'influence':
        reliability = influence(ranks, aggregation_type=aggregation_type, top_k=top_k)
    elif metric == 'proximity': 
        reliability = proximity(ranks, n_neighbors=n_neighbors, top_k=top_k)
    elif metric == 'pagerank':
        reliability = pagerank(ranks, top_k=top_k)
    elif metric == 'averagedistance':
        reliability = averagedistance(ranks, top_k=top_k)
    return reliability

def _get_trimmed_ranks_clustering(ranks, reliability):
    clustering = AgglomerativeClustering(n_clusters=2, linkage='single').fit_predict(reliability.reshape((-1, 1)))

    cluster_ids, counts = np.unique(clustering, return_counts=True)
    largest_cluster_idx = cluster_ids[np.argmax(counts)] # Largest cluster
    
    most_reliable_cluster_idx = np.argmax([
        np.sum(reliability[np.where(clustering == 0)[0]]), 
        np.sum(reliability[np.where(clustering == 1)[0]])]) 
        # np.sum(reliability[np.where(clustering == 2)[0]])])

    # trimmed_ranks = ranks[np.where(clustering == largest_cluster_idx)[0], :]
    trimmed_ranks = ranks[np.where(clustering == most_reliable_cluster_idx)[0], :] # <--- NOTE: We used this
    # trimmed_ranks = ranks[reliability > 0, :]
    
    return trimmed_ranks

def compute_weights(ranks:np.ndarray, 
                    true_rank:Optional[np.ndarray]=None)->np.ndarray:
    """Computes the weight of a data point based on its distance from the true permutation. 
    """
    n_metrics, n_models = ranks.shape
    if true_rank is None: true_rank = np.arange(n_models)
    distance_from_true_rank = np.array([mk.distance(perm, true_rank) for perm in ranks])
    scaler = MinMaxScaler(feature_range=(0, 9))
    weights = scaler.fit_transform(distance_from_true_rank.reshape((-1, 1)))
    weights = 1/(1 + weights)
    return weights.reshape((-1, 1))

def build_graph(ranks:np.ndarray, 
                metric_weights:Optional[np.ndarray]=None)->np.ndarray:
    n_metrics, n_models = ranks.shape
    if metric_weights is None: 
        metric_weights = np.ones((n_metrics, 1))
    else: 
        metric_weights = metric_weights.reshape((-1, 1))
    edge_weights = np.zeros((n_models, n_models))
    
    for i, j in combinations(range(n_models), 2):
        preference = ranks[:, i] - ranks[:, j]
        h_ij = (metric_weights.T @ (preference < 0).astype(int).reshape((-1, 1))).squeeze()  # prefers i to j
        h_ji = (metric_weights.T @ (preference > 0).astype(int).reshape((-1, 1))).squeeze()  # prefers j to i
        if h_ij > h_ji:
            edge_weights[i, j] = h_ij - h_ji
        elif h_ij < h_ji:
            edge_weights[j, i] = h_ji - h_ij
    return edge_weights    

##########################################
# Functions to compute influence
##########################################

def objective(ranks, aggregation_type='kemeny', top_k=None):
    if aggregation_type == 'borda':
        _, sigma_star = borda(ranks=ranks, weights=None)
    elif aggregation_type == 'kemeny':
        _, sigma_star = kemeny(ranks=ranks, weights=None, verbose=False)
    elif aggregation_type == 'partial_borda':
        _, sigma_star = partial_borda(ranks=ranks, weights=None, top_k=top_k)
    return np.mean([mk.distance(r, sigma_star) for r in ranks])
    
def influence(ranks, aggregation_type='kemeny', top_k=None)->np.array:
    """Computes the reciprocal influence of each permutation/rank on the objective. Ranks with 
    higher influence (and lower reciprocal influence) are more outlying.
    """ 
    N, n = ranks.shape
    objective_values = []
    tol = 1e-6

    if (aggregation_type=='partial_borda') and (top_k is None):
        raise ValueError("top_k must be specified!")
    
    objective_all = objective(ranks, aggregation_type=aggregation_type, top_k=top_k) # Objective when using all the permutations
    
    for i in combinations(np.arange(N), N-1):
        objective_values.append( objective(ranks[i, :], aggregation_type=aggregation_type, top_k=top_k) ) 
    
    # If removing a permutation results in a higher decrease in the objective
    # then it is more likely to be outlying
    influence = objective_all - np.array(objective_values[::-1]) # Reverse the list
    reliability = -influence

    # influence -- 
    # +ve -- metric good
    # -ve influence is bad
    # low positive influence or high positive influence?

    return reliability

def proximity(ranks, n_neighbors:int=6, top_k=None)->np.array:
    """Computes the proximity of each rank to its nearest neighbours. Ranks with higher proximity are more central. 
    """
    if top_k is not None: 
        ranks = ranks.astype(float)
        x, y = np.where((ranks > (top_k-1)))
        for x_i, y_i in zip(x, y): ranks[x_i, y_i] = np.NaN

    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric=mk.distance)
    neigh.fit(ranks)

    proximity = 1/neigh.kneighbors(ranks)[0].mean(axis=1)

    return proximity

def pagerank(ranks, top_k=None)->np.array:
    """Computes the pagerank of each rank. Higher pagerank implies that a rank is more authoritative.
    """
    if top_k is not None: 
        ranks = ranks.astype(float)
        x, y = np.where((ranks > (top_k-1)))
        for x_i, y_i in zip(x, y): ranks[x_i, y_i] = np.NaN

    G = nx.Graph()

    # Create weighted undirected graph
    pdistmatrix = pdist(ranks, metric=mk.distance)
    m, _ = ranks.shape

    elist = []
    for i, j in combinations(np.arange(m), r=2):
        idx = m * i + j - ((i + 2) * (i + 1)) // 2
        elist.append( (i, j, 1/(1+pdistmatrix[idx])) )   

    G.add_weighted_edges_from(elist)  
    
    # Compute the pagerank of each node
    pagerank = np.array(list(nx.pagerank(G).values()))
    
    return pagerank

def averagedistance(ranks, top_k=None)->np.array:
    """Computes the average distance of each rank to all other ranks. 
    Lower average implies that a rank is more reliable.
    """
    if top_k is not None: 
        ranks = ranks.astype(float)
        x, y = np.where((ranks > (top_k-1)))
        for x_i, y_i in zip(x, y): ranks[x_i, y_i] = np.NaN
    
    tol = 1e-6
    averagedist = squareform(pdist(ranks, metric=mk.distance)).mean(axis=1)
    return 1/(tol+averagedist)