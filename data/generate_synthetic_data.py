#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
## Code to generate synthetic time series
def basic_ts(type, freq, amplitude, n_time, stop, eps_std):
    x = np.linspace(start=0, stop=stop, num=n_time)
    if type=='sin':
        y = amplitude*np.sin(freq * x)
    elif type=='cos':
        y = amplitude*np.cos(freq * x)
    eps  = np.random.normal(loc=0.0, scale=eps_std, size=n_time)
    return y, eps, x
    
def multivariate_ts(n_series, n_time, eps_std, stop):
    Y = []
    E = []
    freqs = np.random.randint(5, 50, size=n_series) # Parameterise 5, 50 such that 
    amplitudes = np.random.rand(n_series)
    for idx in range(n_series):
        type = 'sin' if np.random.rand() < 0.5 else 'cos'
        y, eps, _ = basic_ts(type=type, freq=freqs[idx], amplitude=amplitudes[idx],
                             stop=stop, n_time=n_time, eps_std=eps_std)
        Y.append(y[None,:])
        E.append(eps[None,:])
    Y = np.concatenate(Y)
    E = np.concatenate(E)
    Y = Y + E
    return Y