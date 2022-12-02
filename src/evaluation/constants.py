#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

MODEL_NAMES = [
    'RNN_1', 'RNN_2', 'RNN_3', 'RNN_4', 'LSTMVAE_1', 'LSTMVAE_2', 'LSTMVAE_3', 'LSTMVAE_4', 'NN_1', 'NN_2', 'NN_3', 'DGHL_1', 'DGHL_2', 'DGHL_3', 'DGHL_4', 'MD_1', 'RM_1', 'RM_2', 'RM_3'
]

ANOMALY_TYPES = [
    'average', 'contextual', 'cutoff', 'flip', 'noise', 'scale', 'speedup', 'wander', 'spikes'
]

# Constituents of synthetic predictions
QUANTITIES_SYNTHETIC_PREDICTIONS = [
    'anomalysizes_type_spikes', 
    'anomalylabels_type_spikes',
    'entityscores_type_spikes', 
    'Ta_type_spikes',
    'anomalysizes_type_contextual', 
    'anomalylabels_type_contextual',
    'entityscores_type_contextual', 
    'Ta_type_contextual',
    'anomalysizes_type_flip', 
    'anomalylabels_type_flip',
    'entityscores_type_flip', 
    'Ta_type_flip', 
    'anomalysizes_type_speedup',
    'anomalylabels_type_speedup', 
    'entityscores_type_speedup',
    'Ta_type_speedup', 
    'anomalysizes_type_noise', 
    'anomalylabels_type_noise',
    'entityscores_type_noise', 
    'Ta_type_noise', 
    'anomalysizes_type_cutoff',
    'anomalylabels_type_cutoff', 
    'entityscores_type_cutoff', 
    'Ta_type_cutoff',
    'anomalysizes_type_scale', 
    'anomalylabels_type_scale',
    'entityscores_type_scale', 
    'Ta_type_scale', 
    'anomalysizes_type_wander',
    'anomalylabels_type_wander', 
    'entityscores_type_wander', 
    'Ta_type_wander',
    'anomalysizes_type_average', 
    'anomalylabels_type_average',
    'entityscores_type_average', 
    'Ta_type_average'
]
# Constituents of model predictions 
QUANTITIES_PREDICTIONS = [
    'entity_scores', 'Y', 'Y_hat', 'Y_sigma', 'anomaly_labels', 'mask'
]
N_MODELS = 19
N_ANOMALY_TYPES = 9
N_SYNTHETIC_QUANTITIES = 36
N_QUANTITIES = 6