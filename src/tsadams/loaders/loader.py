#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch as t

from ..datasets.dataset import Dataset, Entity

from typing import List, Tuple, Optional, Union
#TODO: add label in batch
#TODO: ds
#TODO" print

class Loader(object):
    def __init__(self,
                 dataset: Union[Dataset, Entity],
                 batch_size: int,
                 window_size: int,
                 window_step: int,
                 shuffle: bool = True,
                 padding_type: str = 'None',
                 sample_with_replace: bool = False,
                 verbose: bool = False, 
                 mask_position: str = 'None',
                 n_masked_timesteps: int = 0) -> 'Loader':
        """
        Parameters
        ----------
        dataset: Dataset object
            Dataset to sample windows.
        batch_size: int
            Batch size.
        windows_size: int
            Size of windows to sample.
        window_step: int
            Step size between windows.
        shuffle: bool
            Shuffle windows.
        padding_type: str
            Pad initial, last or None window with 0s.
        sample_with_replace: bool
            When shuffling, sample windows with replacement. When true, behaviour is equivalent to train with iterations
            instead of epochs.
        verbose:
            Boolean for printing details.
        mask_position: str 
            Position of timesteps to mask. Can be one of 'None', 'right', 'mid'
            NOTE: Currently we will only support masking all the features at a masked timesteps. 
            TODO: Support masking some features in some timesteps.  
        n_masked_timesteps: int
            Number of timesteps to be masked
            NOTE: Currently we will only support single step masking. 
        """
        if isinstance(dataset, Entity):
            dataset = Dataset(entities=[dataset], name=dataset.name, verbose=False)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.padding_type = padding_type
        self.sample_with_replace = sample_with_replace
        self.verbose = verbose
        self.mask_position = mask_position
        self.n_masked_timesteps = n_masked_timesteps

        _VALID_MASK_POSITIONS = ['None', 'right', 'mid']
        if mask_position not in _VALID_MASK_POSITIONS: 
            raise ValueError(f'mask_position must be one of {_VALID_MASK_POSITIONS}, {mask_position} was passed.')

        if window_size > 0:
            self.window_size = window_size
            self.window_step = window_step
        else:
            self.window_size = dataset.total_time # TODO: will only work with 1 entity
            self.window_step = dataset.total_time
            self.padding_type = 'None'

        self._create_windows()


    def _array_to_windows(self, X):
        """
        """
        windows = t.Tensor(X)

        # Padding
        _, n_time = X.shape

        if (self.window_size > 0) and (self.padding_type != 'None'):
            self.padding_size = self.window_step - (n_time - self.window_size) % self.window_step
            left_padding = self.padding_size if self.padding_type == 'left' else 0
            right_padding = self.padding_size if self.padding_type == 'right' else 0

        else:
            self.padding_size = 0
            left_padding = 0
            right_padding = 0
        
        padder = t.nn.ConstantPad1d(padding=(left_padding, right_padding), value=0)
        windows = padder(windows)

        # Creating rolling windows and 'flattens' them
        windows = windows.unfold(dimension=-1, 
                                 size=self.window_size, 
                                 step=self.window_step)

        windows = windows.permute(1, 0, 2)
        return windows

    def _create_windows(self):
        """
        """
        self.Y_windows = []
        self.mask_windows = []
        self.X_windows = []
        for entity in self.dataset.entities:
            self.Y_windows.append(self._array_to_windows(entity.Y))
            self.mask_windows.append(self._array_to_windows(entity.mask))
            if self.dataset.n_exogenous:
                self.X_windows.append(self._array_to_windows(entity.X))

        self.Y_windows = t.cat(self.Y_windows)
        self.mask_windows = t.cat(self.mask_windows)
        if self.dataset.n_exogenous:
            self.X_windows = t.cat(self.X_windows)
        else:
            self.X_windows = None
        self.n_idxs = len(self.Y_windows)
        self.n_batch_in_epochs = int(np.ceil(self.n_idxs / self.batch_size))

    def __len__(self):
        return self.n_idxs

    def __iter__(self):
        if self.shuffle:
            sample_idxs = np.random.choice(a=self.n_idxs,
                                           size=self.n_idxs,
                                           replace=self.sample_with_replace)
        else:
            sample_idxs = np.arange(self.n_idxs)

        for idx in range(self.n_batch_in_epochs):
            batch_idx = sample_idxs[(idx * self.batch_size) : (idx + 1) * self.batch_size]
            batch = self.__get_item__(idx=batch_idx)
            yield batch

    def apply_mask(self, Y_batch): 
        mask = np.ones(Y_batch.shape)
        if (self.n_masked_timesteps == 0) or ('None' in self.mask_position):
            # Then no masking
            return Y_batch, mask
        else: 
            if self.mask_position == 'right':
                mask_idx = -self.n_masked_timesteps - 1
                Y_batch[:, :, (mask_idx+1):] = 0
                mask[:, :, (mask_idx+1):] = 0
            elif self.mask_position == 'mid':
                mask_idx = np.random.randint(Y_batch.shape[2] - self.n_masked_timesteps)
                Y_batch[:, :, (mask_idx+1):(mask_idx + self.n_masked_timesteps)] = 0
                mask[:, :, (mask_idx+1):(mask_idx + self.n_masked_timesteps)] = 0
            return Y_batch, mask

    def __get_item__(self, idx):
        """
        """
        # Index windows from tensors
        Y_batch = self.Y_windows[idx]
        masked_Y_batch, mask = self.apply_mask(Y_batch)
        mask_batch = self.mask_windows[idx].numpy()
        mask_batch = np.logical_and(mask_batch, mask).astype(int)
        if self.X_windows is None:
            X_batch = []
        else:
            X_batch = self.X_windows[idx]
        
        # Batch
        batch = {'Y': t.as_tensor(masked_Y_batch),
                 'mask': t.as_tensor(mask_batch),
                 'X': t.as_tensor(X_batch),
                 'idx': idx}

        return batch

    def __str__(self):
        # TODO: Complete the loader. 
        return 'I am a loader'
