#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Anomaly parameters
#######################################

ANOMALY_PARAM_GRID = {
    'spikes': {
        'anomaly_type': ['spikes'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
        'anomaly_propensity': [0.5],
    },
    'contextual': {
        'anomaly_type': ['contextual'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
    },
    'flip': {
        'anomaly_type': ['flip'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
    },
    'speedup': {
        'anomaly_type': ['speedup'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'speed': [0.25, 0.5, 2, 4],
        'scale': [2],
    },
    'noise': {
        'anomaly_type': ['noise'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'noise_std': [0.05],
        'scale': [2],
    },
    'cutoff': {
        'anomaly_type': ['cutoff'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'constant_type': ['noisy_0', 'noisy_1'],
        'constant_quantile': [0.75],
        'scale': [2],
    },
    'scale': {
        'anomaly_type': ['scale'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'amplitude_scaling': [0.25, 0.5, 2, 4],
        'scale': [2],
    },
    'wander': {
        'anomaly_type': ['wander'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'baseline': [-0.3, -0.1, 0.1, 0.3],
        'scale': [2],
    },
    'average': {
        'anomaly_type': ['average'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'ma_window': [4, 8],
        'scale': [2],
    }
}
