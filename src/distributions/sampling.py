#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Optional, Tuple
import distributions.mallows_kendall as mk
from distributions.pl_model import PlackettLuce


def sample_random(N: int = 20, n: int = 100, seed: int = 0) -> np.ndarray:
    """This function generates m random permutations of size n.
        Parameters
        ----------
        N: int
            Number of permutations to generate
        n: int
            Length of permutation (number of items)
        seed: float
            Random Seed
        
        Returns
        -------
        ndarray
            The rankings generated
    """
    np.random.seed(seed)
    return np.array([np.random.permutation(n) for i in range(N)])


def sample_mallows(N: int = 20,
                   n: int = 100,
                   theta: float = 0.1,
                   s0: Optional[np.ndarray] = None) -> np.ndarray:
    if s0 is None:
        s0 = np.array(range(n))
    return mk.sample(m=N, n=n, theta=theta, s0=s0)


def sample_pl(N: int = 20,
              n: int = 100,
              scores: np.ndarray = np.arange(1, 101),
              seed: int = 0) -> np.ndarray:
    dist = PlackettLuce(scores, seed)
    return dist.sample(size=N).numpy()


def sample_pl_with_noise(N: int = 20,
                         n: int = 100,
                         scores: np.ndarray = np.arange(1, 101),
                         noise: float = 0.1,
                         seed: int = 0) -> np.ndarray:
    """Generates samples from the PL distribution with some noise. 

    Parameters
    ----------
    N: int
        Number of permutations
    n: int
        Number of items
    scores: np.ndarray
        Scores of ranks.
    noise: float
        Percentage of noisy permutations
    seed: float
        Random Seed
    """

    n_outsample = int(noise * N)
    n_insample = N - n_outsample

    ranks_pl = sample_pl(N=n_insample, n=n, scores=scores, seed=seed)
    ranks_random = sample_random(N=n_outsample, n=n)

    if noise > 0:
        ranks = np.concatenate([ranks_pl, ranks_random], axis=0)
    else:
        ranks = ranks_pl

    return ranks, n_outsample, n_insample


def sample_mallows_with_noise(N: int = 20,
                              n: int = 100,
                              type: str = 'random',
                              theta: float = 0.1,
                              scale: float = 100,
                              noise: float = 0.1,
                              seed: int = 0) -> np.ndarray:
    """Generates samples from the mallows distribution with some noise. 

    Parameters
    ----------
    N: int
        Number of permutations
    n: int
        Number of items
    type: str
        Type of noise. Can be one of 'random' or 'mixture'.
        When type is set to 'random' we draw noisy samples as
        random permutations. In case of mixtures, we draw a 
        noisy samples from a mallows distribution centered at 
        the indentity permutation but with 100 times lower 
        dispersion. 
    scale: float
        Only matters when type is set to mixture. The dispersion 
        parameter of the noisy mallows distribution is set to theta/scale. 
    theta: float
        Dispersion parameter of Mallows
    noise: float
        Percentage of noisy permutations
    seed: float
        Random Seed
    """
    np.random.seed(seed)
    n_outsample = int(noise * N)
    n_insample = N - n_outsample

    ranks_mallow = sample_mallows(N=n_insample, n=n, theta=theta)
    if type == 'random':
        ranks_random = sample_random(N=n_outsample, n=n)
    elif type == 'mixture':
        ranks_random = sample_mallows(N=n_outsample, n=n, theta=theta / scale)
    elif type == 'correlated':
        ranks_random = sample_mallows(N=n_outsample,
                                      n=n,
                                      theta=theta / scale,
                                      s0=np.random.permutation(n))

    if noise > 0:
        ranks = np.concatenate([ranks_mallow, ranks_random], axis=0)
    else:
        ranks = ranks_mallow

    return ranks, n_outsample, n_insample
