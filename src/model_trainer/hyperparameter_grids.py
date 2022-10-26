#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# DGHL Model hyper-parameter grid
#######################################

DGHL_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'train_batch_size': [64],
    'learning_rate': [1e-3],
    'seed': [1],
    'max_steps': [1000],
    'eval_batch_size': [128],
}

DGHL_PARAM_GRID = {
    'window_size': [64],
    'window_step': [64],
    'hidden_multiplier': [32],
    'max_filters': [256],
    'kernel_multiplier': [1],
    'a_L': [1],  # Sub-windows [1, 4]
    'z_size': [25, 50],  # Size of latent z vector [5, 25, 50]
    'z_size_up': [5],
    'z_iters': [
        25, 100
    ],  # Number of iteration in the Langevyn dynamics inference formula. [5, 25, 100] -- more the better and slower. Linear time dependence. 
    'z_iters_inference': [100],  # Higher the better -> 300, 500 better. 
    'z_sigma': [0.25],
    'z_step_size': [0.1],
    'z_with_noise': [False],
    'z_persistent': [
        False
    ],  # Can only be False currently. = True means that it will start from the last latent vector when it observed the particular window. Therefore it needs higher z_iters right now. 
    'normalize_windows': [True],
    'noise_std': [0.001],
    'random_seed': [1],
    'device': [None]
}

#######################################
# Running Mean Model hyper-parameter grid
#######################################

RM_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [1],
}

RM_PARAM_GRID = {
    'window_size': [-1],  # Works with the entire time series 
    'window_step': [-1],
    'running_window_size': [4, 16, 64],
    'device': [None]
}

#######################################
# Mean Deviation Model hyper-parameter grid
#######################################

MD_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'train_batch_size': [256],
    'learning_rate': [1e-3],
    'seed': [1],
    'max_steps': [5000],
    'eval_batch_size': [1],
}

MD_PARAM_GRID = {
    'window_size': [64],  # Works with the entire time series 
    'window_step': [64],
    'device': [None]
}

#######################################
# Nearest Neighbours model hyper-parameter grid
#######################################

NN_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

NN_PARAM_GRID = {
    'window_size': [64],
    'window_step': [64],
    'n_neighbors': [1, 3, 5]
}

#######################################
# LSTM-VAE model hyper-parameter grid
#######################################

LSTMVAE_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'train_batch_size': [256],
    'learning_rate': [0.0005],
    'seed': [1],
    'max_steps': [1000],
    'eval_batch_size': [128],
}

LSTMVAE_PARAM_GRID = {
    'window_size': [64],
    'window_step': [64],
    'hidden_size':
    [512, 256],  # hidden_size – The number of features in the hidden state h
    'latent_size': [256, 128],  # Size of the latent z 
    'num_layers': [
        4
    ],  # Number of recurrent layers. Setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM.
    'noise_std': [0.001],
    'random_seed': [1],
    'device': [None]
}

#######################################
# RNN model hyper-parameter grid
#######################################

RNN_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'train_batch_size': [1],
    'learning_rate': [0.01],
    'seed': [1],
    'max_steps': [100],
    'eval_batch_size': [1],
}

RNN_PARAM_GRID = {
    'window_size': [-1],
    'window_step': [-1],
    'input_size': [32, 64],
    'output_size': [8],
    'sample_freq': [8],
    'n_t': [0],
    'cell_type': ['LSTM'],
    'dilations': [[[1, 2], [4, 8]]],
    'state_hsize': [128, 256],  # 128
    'add_nl_layer': [False],
    'random_seed': [1],
    'device': [None]
}
