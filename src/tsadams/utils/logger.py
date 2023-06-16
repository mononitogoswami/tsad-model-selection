#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional, Union, Dict
import pandas as pd
import pickle as pkl
import numpy as np
import torch as t

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
