#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Optional, Union, Dict
import numpy as np
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import pickle as pkl
from re import L
import torch as t
import os
import matplotlib.pyplot as plt


def visualize_predictions(predictions: dict, savefig=True):
    """Visualizes univariate models given the predictions dictionary
    """
    MODEL_NAMES = list(predictions.keys())
    fig, axes = plt.subplots(len(MODEL_NAMES),
                             1,
                             sharey=True,
                             sharex=True,
                             figsize=(30, 5 * len(MODEL_NAMES)))

    for i, model_name in enumerate(MODEL_NAMES):
        start_anomaly = np.argmax(
            np.diff(predictions[model_name]['anomaly_labels'].flatten()))
        end_anomaly = np.argmin(
            np.diff(predictions[model_name]['anomaly_labels'].flatten()))
        axes[i].plot(predictions[model_name]['Y'].flatten(),
                     color='darkblue',
                     label='Y')
        axes[i].plot(predictions[model_name]['Y_hat'].flatten(),
                     color='darkgreen',
                     label='Y_hat')
        axes[i].plot(
            np.arange(start_anomaly, end_anomaly),
            predictions[model_name]['Y'].flatten()[start_anomaly:end_anomaly],
            color='red',
            label='Anomaly')

        entity_scores = predictions[model_name]['entity_scores'].flatten()
        entity_scores = (entity_scores - entity_scores.min()) / (
            entity_scores.max() - entity_scores.min())
        # entity_scores = (entity_scores - entity_scores.mean())/(entity_scores.std())
        axes[i].plot(entity_scores,
                     color='magenta',
                     linestyle='--',
                     label='Anomaly Scores')

        axes[i].set_title(f'Predictions of Model {model_name}', fontsize=16)
        axes[i].legend(fontsize=16, ncol=2, shadow=True, fancybox=True)
        axes[i].set_xlabel('Time', fontsize=16)
        axes[i].set_ylabel('Y', fontsize=16)

    if savefig:
        plt.savefig('predictions.pdf')
    plt.show()


def visualize_data(train_data, test_data, savefig=False):
    """Visualizes train and testing splits of a univariate entity.
    """
    # Visualize the train and the test data
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(25, 4))
    axes[0].plot(train_data.entities[0].Y.flatten(), color='darkblue')
    axes[0].set_title('Train data', fontsize=16)

    start_anomaly = np.argmax(np.diff(test_data.entities[0].labels.flatten()))
    end_anomaly = np.argmin(np.diff(test_data.entities[0].labels.flatten()))

    axes[1].plot(test_data.entities[0].Y.flatten(),
                 color='darkblue',
                 label='Y')
    axes[1].plot(np.arange(start_anomaly, end_anomaly),
                 test_data.entities[0].Y.flatten()[start_anomaly:end_anomaly],
                 color='red',
                 label='Anomaly')
    axes[1].set_title('Test data', fontsize=16)
    axes[1].legend(fontsize=16, ncol=2, shadow=True, fancybox=True)

    if savefig:
        plt.savefig('data_visual.pdf')
    plt.show()


def de_unfold(windows, window_step):
    """Stiches multiple windows together.
    windows of shape (n_windows, n_channels, window_size)
    """
    n_windows, n_channels, window_size = windows.shape

    if window_step < 0:
        window_step = window_size

    assert window_step <= window_size, 'Window step must be smaller than window_size'

    total_len = (n_windows) * window_step + (window_size - window_step)

    x = np.zeros((n_channels, total_len))
    counter = np.zeros((1, total_len))

    for i in range(n_windows):
        start = i * window_step
        end = start + window_size
        x[:, start:end] += windows[i]
        counter[:, start:end] += 1

    x = x / counter

    return x


class Logger(object):

    def __init__(self,
                 save_dir: str = 'results/',
                 overwrite: bool = False,
                 verbose: bool = False) -> 'Logger':
        """
        Parameters
        ----------
       
        save_dir: str
            Path of the directory to save the object
        overwrite: bool
            If file already exists, then raises OSError. If True, overwrites the file. 
        verbose:
            Boolean for printing details.
        """
        self.verbose = verbose
        self.save_dir = save_dir
        self.overwrite = overwrite
        self._VALID_FILE_TYPES = ['auto', 'data', 'torch', 'csv']

    def make_directories(self):
        if self.obj_class is not None:
            if not os.path.exists(self.obj_save_path):
                os.makedirs(self.obj_save_path, exist_ok=True)

    def check_file_exists(self, obj_class: Union[str, List[str]],
                          obj_name: str):
        obj_save_path = self.get_obj_save_path(obj_class)
        return os.path.exists(
            os.path.join(obj_save_path,
                         str(obj_name) + '.pth')) or os.path.exists(
                             os.path.join(
                                 obj_save_path,
                                 str(obj_name) + '.data')) or os.path.exists(
                                     os.path.join(obj_save_path,
                                                  str(obj_name) + '.csv'))

    def save_torch_model(self):
        with open(
                os.path.join(self.obj_save_path,
                             str(self.obj_name) + '.pth'), 'wb') as f:
            # t.save(self.obj.state_dict(), f) # TODO: We will not support saving state dicts for now
            t.save(self.obj, f)

    def save_data_object(self):
        with open(
                os.path.join(self.obj_save_path,
                             str(self.obj_name) + '.data'), 'wb') as f:
            pkl.dump(self.obj, f)

    def save_csv(self):
        self.obj.to_csv(os.path.join(self.obj_save_path,
                                     str(self.obj_name) + '.csv'),
                        index=None)

    def save_meta_object(self):
        with open(
                os.path.join(self.obj_save_path,
                             str(self.obj_name) + '.meta'), 'wb') as f:
            pkl.dump(self.obj_meta, f)

    def get_obj_save_path(self, obj_class):
        return os.path.join(self.save_dir, '/'.join(obj_class))

    def save(self,
             obj: Union[t.nn.Module, np.ndarray, List, Dict],
             obj_name: str,
             obj_meta: Optional[Union[str, List[str]]],
             obj_class: Optional[Union[str, List[str]]],
             type: str = 'auto'):
        """
        Parameters
        ----------
        obj: Union[t.nn.Module, List, np.ndarray]
            Object to save
        obj_name: str
            Name of the object to save
        obj_meta: Optional[str, List[str]]
            Object meta data. 
        obj_class: Optional[str, List[str]]
            Objects can be organised in hierarichal classes.
        type: str
            Filetype for object
        """
        self.obj = obj
        self.obj_class = obj_class
        self.obj_name = obj_name
        self.obj_meta = obj_meta

        self.obj_save_path = self.get_obj_save_path(obj_class)
        self.make_directories(
        )  # Make all necessary directories to save the object

        if not self.overwrite:
            if self.check_file_exists(
                    obj_class=obj_class,
                    obj_name=obj_name):  # Check if the file already exists!
                print(
                    f'File already exists! Overwriting is set to {self.overwrite}'
                )
                return

        if type == 'auto':
            if isinstance(obj, t.nn.Module):
                self.save_torch_model()
            elif isinstance(obj, pd.DataFrame):
                self.save_csv()
            else:
                self.save_data_object()
        elif type == 'torch':
            self.save_torch_model()
        elif type == 'data':
            self.save_data_object()
        elif type == 'csv':
            self.save_csv()

        if self.obj_meta is not None:
            self.save_meta_object()

        if self.verbose:
            print(
                f'Saving file {os.path.join(self.obj_save_path, str(self.obj_name))}'
            )
