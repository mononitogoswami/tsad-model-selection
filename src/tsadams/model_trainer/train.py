#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to train models on a dataset of entities
#######################################

from typing import List, Union, Optional
from sklearn.model_selection import ParameterGrid

from tqdm import tqdm

from .hyperparameter_grids import *  # DGHL_TRAIN_PARAM_GRID, DGHL_PARAM_GRID, MD_TRAIN_PARAM_GRID, MD_PARAM_GRID, RM_PARAM_GRID, RM_TRAIN_PARAM_GRID, NN_PARAM_GRID, NN_TRAIN_PARAM_GRID, LSTMVAE_TRAIN_PARAM_GRID, LSTMVAE_PARAM_GRID, RNN_TRAIN_PARAM_GRID, RNN_PARAM_GRID
from .training_args import TrainingArguments
from .trainer import Trainer

from ..utils.logger import Logger
from ..datasets.load import load_data
from ..loaders.loader import Loader
# Import all the models here!
from ..models.dghl import DGHL
from ..models.rnn import RNN
from ..models.lstmvae import LSTMVAE
from ..models.nearest_neighbors import NearestNeighbors
from ..models.mean_deviation import MeanDeviation
from ..models.running_mean import RunningMean

class TrainModels(object):
    """Class to pre-train models on a dataset/entity.
    
    Parameters
    ----------
    dataset: str
        Name of dataset in which the entity belongs. 
    entity: Union[Dataset, Entity]
        Multivariate timeseries entity on which we need to evaluate performance of models. 

    downsampling: int

    root_dir: str

    batch_size: int 
        Batch size for evaluation
    training_size: float
        Percentage of training data to use for training models. 
    overwrite: bool
        Whether to re-train existing models. 
    verbose: bool
        Controls verbosity
    save_dir: str
        Directory to save the trained models. 
    """

    def __init__(self,
                 dataset: str = 'anomaly_archive',
                 entity: str = '233_UCR_Anomaly_mit14157longtermecg',
                 downsampling: Optional[int] = None,
                 min_length: Optional[int] = None,
                 root_dir: str = '../../datasets/',
                 training_size=1,
                 overwrite: bool = False,
                 verbose: bool = True,
                 save_dir: str = '../trained_models'):

        if training_size > 1.0:
            raise ValueError('Training size must be <= 1.0')
        self.verbose = verbose
        self.train_data = load_data(dataset=dataset,
                                    group='train',
                                    entities=entity,
                                    downsampling=downsampling,
                                    min_length=min_length,
                                    root_dir=root_dir,
                                    verbose=False)

        if verbose:
            print(f'Number of entities: {self.train_data.n_entities}')
            print(
                f'Using the first {training_size*100}\% of the training data to train the models.'
            )

        self.overwrite = overwrite

        for e_i in range(self.train_data.n_entities):
            t = self.train_data.entities[e_i].Y.shape[1]
            self.train_data.entities[e_i].Y = self.train_data.entities[
                e_i].Y[:, :int(training_size * t)]

        # Logger object to save the models
        self.logging_obj = Logger(save_dir=save_dir,
                                  overwrite=self.overwrite,
                                  verbose=verbose)
        self.logging_hierarchy = [dataset, entity]

        self._VALID_MODEL_ARCHITECTURES = [
            'DGHL', 'RNN', 'LSTMVAE', 'NN', 'MD', 'RM'
        ]

    def train_models(self, model_architectures: List[str] = 'all'):
        """Function to selected models. 
        """
        if model_architectures == 'all':
            model_architectures = self._VALID_MODEL_ARCHITECTURES

        for model_name in model_architectures:
            if 'DGHL' == model_name:
                self.train_dghl()
            elif 'RNN' == model_name:
                self.train_rnn()
            elif 'LSTMVAE' == model_name:
                self.train_lstmvae()
            elif 'NN' == model_name:
                self.train_nn()
            elif 'MD' == model_name:
                self.train_md()
            elif 'RM' == model_name:
                self.train_rm()

    def train_dghl(self):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(DGHL_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(DGHL_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                if self.train_data.entities[
                        0].X is not None:  # DGHL also considers covariates
                    model_hyper_params[
                        'n_features'] = self.train_data.n_features + self.train_data.entities[
                            0].X.shape[0]
                else:
                    model_hyper_params[
                        'n_features'] = self.train_data.n_features

                training_args = TrainingArguments(**train_hyper_params)
                model = DGHL(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"DGHL_{MODEL_ID+1}"):
                        print(f'Model DGHL_{MODEL_ID+1} already trained!')
                        continue

                trainer = Trainer(model=model,
                                  args=training_args,
                                  train_dataset=self.train_data,
                                  eval_dataset=None,
                                  verbose=self.verbose)
                trainer.train()
                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=trainer.model,
                                      obj_name=f"DGHL_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_md(self):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(MD_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(MD_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model_hyper_params['n_features'] = self.train_data.n_features
                training_args = TrainingArguments(**train_hyper_params)
                model = MeanDeviation(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"MD_{MODEL_ID+1}"):
                        print(f'Model MD_{MODEL_ID+1} already trained!')
                        continue

                trainer = Trainer(model=model,
                                  args=training_args,
                                  train_dataset=self.train_data,
                                  eval_dataset=None,
                                  verbose=self.verbose)
                trainer.train()
                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=trainer.model,
                                      obj_name=f"MD_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_lstmvae(self):
        MODEL_ID = 0
        model_hyper_param_configurations = list(
            ParameterGrid(LSTMVAE_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(LSTMVAE_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model_hyper_params['n_features'] = self.train_data.n_features
                training_args = TrainingArguments(**train_hyper_params)
                model = LSTMVAE(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"LSTMVAE_{MODEL_ID+1}"):
                        print(f'Model LSTMVAE_{MODEL_ID+1} already trained!')
                        continue

                trainer = Trainer(model=model,
                                  args=training_args,
                                  train_dataset=self.train_data,
                                  eval_dataset=None,
                                  verbose=self.verbose)
                trainer.train()
                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=trainer.model,
                                      obj_name=f"LSTMVAE_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_rnn(self):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(RNN_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(RNN_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                training_args = TrainingArguments(**train_hyper_params)
                model = RNN(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"RNN_{MODEL_ID+1}"):
                        print(f'Model RNN_{MODEL_ID+1} already trained!')
                        continue

                trainer = Trainer(model=model,
                                  args=training_args,
                                  train_dataset=self.train_data,
                                  eval_dataset=None,
                                  verbose=self.verbose)
                trainer.train()
                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=trainer.model,
                                      obj_name=f"RNN_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_rm(self):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(RM_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(RM_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = RunningMean(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"RM_{MODEL_ID+1}"):
                        print(f'Model RM_{MODEL_ID+1} already trained!')
                        continue

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"RM_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_nn(self, batch_size=32):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(NN_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(NN_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = NearestNeighbors(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"NN_{MODEL_ID+1}"):
                        print(f'Model NN_{MODEL_ID+1} already trained!')
                        continue

                dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(dataloader)

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"NN_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)
