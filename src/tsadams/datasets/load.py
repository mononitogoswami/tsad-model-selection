import os
import shutil
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from .dataset import Entity, Dataset
from ..utils.data_utils import download_file
from sklearn.preprocessing import MinMaxScaler


MACHINES = ['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-1-8',
            'machine-2-1', 'machine-2-2','machine-2-3','machine-2-4','machine-2-5','machine-2-6','machine-2-7','machine-2-8','machine-2-9', 
            'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4','machine-3-5','machine-3-6','machine-3-7','machine-3-8', 'machine-3-9',
            'machine-3-10', 'machine-3-11']

# Data URIs
SMD_URL = 'https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset'
NASA_DATA_URI = r'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'
NASA_LABELS_URI = r'https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv'
ANOMALY_ARCHIVE_URI = r'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip'
VALID_DATASETS = ['msl', 'smap', 'smd', 'anomaly_archive', 'swat', 'synthetic']

def load_data(dataset: str, group: str, entities: Union[str, List[str]], downsampling: float=None, min_length: float=None, root_dir:str='./data', normalize:bool=True, verbose:bool=True):
    """Function to load TS anomaly detection datasets.
    Parameters
    ----------
    dataset: str
        Name of the dataset. 
    group: str
        The train or test split. 
    entities: Union[str, List[str]]
        Entities to load from the dataset. 
    downsampling: Optional[float]
        Whether and the extent to downsample the data. 
    root_dir: str
        Path to the directory where the datasets are stored. 
    normalize: bool
        Whether to normalize Y. 
    verbose: bool
        Controls verbosity
    """
    if dataset == 'smd':
        return load_smd(group=group, machines=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose)
    elif dataset == 'msl':
        return load_msl(group=group, channels=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose)
    elif dataset == 'smap':
        return load_smap(group=group, channels=entities, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose)
    elif dataset == 'anomaly_archive':
        return load_anomaly_archive(group=group, datasets=entities, downsampling=downsampling, min_length=min_length, root_dir=root_dir, normalize=normalize, verbose=verbose)
    elif dataset == 'swat':
        raise NotImplementedError()
    elif dataset == 'synthetic':
        raise NotImplementedError()
    else: 
        raise ValueError(f'Dataset must be one of {VALID_DATASETS}, but {dataset} was passed!')

def load_smd(group, machines=None, downsampling=None, root_dir='./data', normalize=True, verbose=True):
    # NOTE: The SMD dataset is normalized and therefore we do not need normalize it further. The normalize parameter is for input compatibility. 
    if machines is None:
        machines = MACHINES

    if isinstance(machines, str): 
        machines = [machines]

    root_dir = f'{root_dir}/ServerMachineDataset'

    # Download data
    for machine in machines:
        
        if not os.path.exists(f'{root_dir}/train/{machine}.txt'):

            download_file(filename=f'{machine}.txt',
                          directory=f'{root_dir}/train',
                          source_url=f'{SMD_URL}/train/{machine}.txt')

            download_file(filename=f'{machine}.txt',
                          directory=f'{root_dir}/test',
                          source_url=f'{SMD_URL}/test/{machine}.txt')

            download_file(filename=f'{machine}.txt',
                          directory=f'{root_dir}/test_label',
                          source_url=f'{SMD_URL}/test_label/{machine}.txt')

    # Load data
    entities = []
    for machine in machines:
        if group=='train':
            name = 'smd-train'
            train_file = f'{root_dir}/train/{machine}.txt'
            Y = np.loadtxt(train_file, delimiter=',').T

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape

                right_padding = downsampling - n_t%downsampling
                Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)

            entity = Entity(Y=Y, name=machine, verbose=verbose)
            entities.append(entity)

        elif group=='test':
            name = 'smd-test'
            test_file = f'{root_dir}/test/{machine}.txt'
            label_file = f'{root_dir}/test_label/{machine}.txt'

            Y = np.loadtxt(test_file, delimiter=',').T
            labels = np.loadtxt(label_file, delimiter=',')

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape
                right_padding = downsampling - n_t%downsampling

                Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                labels = np.pad(labels, (right_padding, 0))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
                labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)
            
            labels = labels[None, :]
            entity = Entity(Y=Y, name=machine, labels=labels, verbose=verbose)
            entities.append(entity)

    smd = Dataset(entities=entities, name=name, verbose=verbose)

    return smd

def download_nasa(root_dir='./data'):
    """Convenience function to download the NASA data
    """
    # Download the data
    download_file(filename=f'NASA',
                 directory=root_dir,
                 source_url=NASA_DATA_URI, 
                 decompress=True)

    # Reorganising the data
    os.remove(os.path.join(root_dir, 'NASA'))# Remove the NASA file
    shutil.rmtree(os.path.join(root_dir, 'data', '2018-05-19_15.00.10'))# Remove unnecessary directories
    shutil.move(src=os.path.join(root_dir, 'data', 'train'), dst=root_dir) # Remove unnessary folder structure
    shutil.move(src=os.path.join(root_dir, 'data', 'test'), dst=root_dir) # Remove unnessary folder structure
    shutil.rmtree(os.path.join(root_dir, 'data'))# Remove unnecessary directories

    # Download the meta-data file
    download_file(filename=f'labeled_anomalies.csv',
                  directory=os.path.join(root_dir),
                  source_url=NASA_LABELS_URI,
                  decompress=False)

def load_msl(group, channels=None, downsampling=None, root_dir='./data', normalize=True, verbose=True):
    return _load_nasa(group=group, spacecraft='MSL', channels=channels, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose)
    
def load_smap(group, channels=None, downsampling=None, root_dir='./data', normalize=True, verbose=True):
    return _load_nasa(group=group, spacecraft='SMAP', channels=channels, downsampling=downsampling, root_dir=root_dir, normalize=normalize, verbose=verbose)

def _load_nasa(group, spacecraft, channels=None, downsampling=None, root_dir='./data', normalize=True, verbose=True):
    root_dir = f'{root_dir}/NASA'
    if not os.path.exists(f'{root_dir}'): download_nasa(root_dir=root_dir)
    meta_data = pd.read_csv(f'{root_dir}/labeled_anomalies.csv')

    CHANNEL_IDS =  list(meta_data.loc[meta_data['spacecraft'] == spacecraft]['chan_id'].values)
    if verbose: 
        print(f'Number of Entities: {len(CHANNEL_IDS)}')

    if channels is None: channels = CHANNEL_IDS

    if isinstance(channels, str): 
        channels = [channels]
    
    entities = []
    for channel_id in channels: 
        if normalize:
            with open(f'{root_dir}/train/{channel_id}.npy', 'rb') as f: 
                Y = np.load(f) # Transpose dataset
            scaler = MinMaxScaler()
            scaler.fit(Y)
            
        if group == 'train':
            name = f'{spacecraft}-train'
            with open(f'{root_dir}/train/{channel_id}.npy', 'rb') as f: 
                Y = np.load(f).T # Transpose dataset

            if normalize: 
                Y = scaler.transform(Y.T).T

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape

                right_padding = downsampling - n_t%downsampling
                Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)

            entity = Entity(Y=Y[0, :].reshape((1, -1)), X=Y[1:, :], name=channel_id, verbose=verbose)
            entities.append(entity)

        elif group == 'test':
            name = f'{spacecraft}-test'
            with open(f'{root_dir}/test/{channel_id}.npy', 'rb') as f: 
                Y = np.load(f).T # Transpose dataset

            if normalize: 
                Y = scaler.transform(Y.T).T

            # Label the data 
            labels = np.zeros(Y.shape[1])
            anomalous_sequences = eval(meta_data.loc[meta_data['chan_id'] == channel_id]['anomaly_sequences'].values[0])
            if verbose: print('Anomalous sequences:', anomalous_sequences)
            
            for interval in anomalous_sequences:    
                labels[interval[0]:interval[1]] = 1

            # Downsampling
            if downsampling is not None:
                n_features, n_t = Y.shape
                right_padding = downsampling - n_t%downsampling

                Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                labels = np.pad(labels, (right_padding, 0))

                Y = Y.reshape(n_features, Y.shape[-1]//downsampling, downsampling).max(axis=2)
                labels = labels.reshape(labels.shape[0]//downsampling, downsampling).max(axis=1)
            
            labels = labels[None, :]
            entity = Entity(Y=Y[0, :].reshape((1, -1)), X=Y[1:, :], name=channel_id, labels=labels, verbose=verbose)
            entities.append(entity)

    data = Dataset(entities=entities, name=name, verbose=verbose)

    return data

def download_anomaly_archive(root_dir='./data'):
    """Convenience function to download the Timeseries Anomaly Archive datasets 
    """
    # Download the data
    download_file(filename=f'AnomalyArchive',
                directory=root_dir,
                source_url=ANOMALY_ARCHIVE_URI, 
                decompress=True)

    # Reorganising the data
    shutil.move(src=f'{root_dir}/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData', 
                dst=root_dir) 
    os.remove(os.path.join(root_dir, 'AnomalyArchive'))
    shutil.rmtree(os.path.join(root_dir, 'AnomalyDatasets_2021'))
    shutil.move(src=f'{root_dir}/UCR_Anomaly_FullData', 
                dst=f'{root_dir}/AnomalyArchive') 

def load_anomaly_archive(group, datasets=None, downsampling=None, min_length=None, root_dir='./data', normalize=True, verbose=True):
    if not os.path.exists(f'{root_dir}/AnomalyArchive/'): download_anomaly_archive(root_dir=root_dir)

    ANOMALY_ARCHIVE_ENTITIES = ['_'.join(e.split('_')[:4]) for e in os.listdir(os.path.join(root_dir, 'AnomalyArchive'))]
    ANOMALY_ARCHIVE_ENTITIES = sorted(ANOMALY_ARCHIVE_ENTITIES)

    if datasets is None: datasets = ANOMALY_ARCHIVE_ENTITIES
    if verbose: print(f'Number of datasets: {len(datasets)}')

    entities = []
    for file in os.listdir(os.path.join(root_dir, 'AnomalyArchive')):

        downsampling_entity = downsampling
        if '_'.join(file.split('_')[:4]) in datasets:
            with open(os.path.join(root_dir, 'AnomalyArchive',  file)) as f:
                Y = f.readlines()
                if len(Y) == 1: 
                    Y = Y[0].strip()
                    Y = np.array([eval(y) for y in Y.split(" ") if len(y) > 1]).reshape((1, -1))
                elif len(Y) > 1: 
                    Y = np.array([eval(y.strip()) for y in Y]).reshape((1, -1))

            fields = file.split('_')
            meta_data = {
                    'name': '_'.join(fields[:4]),
                    'train_end': int(fields[4]),
                    'anomaly_start_in_test': int(fields[5])-int(fields[4]),
                    'anomaly_end_in_test': int(fields[6][:-4])-int(fields[4]),
                }
            if verbose: 
                print(f'Entity meta-data: {meta_data}')

            if normalize:
                Y_train = Y[0, 0:meta_data['train_end']].reshape((-1, 1))
                scaler = MinMaxScaler()
                scaler.fit(Y_train)
                Y = scaler.transform(Y.T).T

            n_time = Y.shape[-1]
            len_train = meta_data['train_end']
            len_test = n_time - len_train

            # No downsampling if n_time < min_length
            if (downsampling_entity is not None) and (min_length is not None):

                if (len_train//downsampling_entity < min_length) or (len_test//downsampling_entity < min_length):
                    downsampling_entity = None

            if group == 'train':
                name = f"{meta_data['name']}-train"
                Y = Y[0, 0:meta_data['train_end']].reshape((1, -1))

                # Downsampling
                if downsampling_entity is not None:
                    n_features, n_t = Y.shape

                    right_padding = downsampling_entity - n_t%downsampling_entity
                    Y = np.pad(Y, ((0,0), (right_padding, 0) ))

                    Y = Y.reshape(n_features, Y.shape[-1]//downsampling_entity, downsampling_entity).max(axis=2)

                entity = Entity(Y=Y.reshape((1, -1)), name=meta_data['name'], verbose=verbose)
                entities.append(entity)

            elif group == 'test':
                name = f"{meta_data['name']}-test"
                Y = Y[0, meta_data['train_end']+1:].reshape((1, -1))

                # Label the data 
                labels = np.zeros(Y.shape[1])
                labels[meta_data['anomaly_start_in_test']:meta_data['anomaly_end_in_test']] = 1            

                # Downsampling
                if downsampling_entity is not None:
                    n_features, n_t = Y.shape
                    right_padding = downsampling_entity - n_t%downsampling_entity

                    Y = np.pad(Y, ((0,0), (right_padding, 0) ))
                    labels = np.pad(labels, (right_padding, 0))

                    Y = Y.reshape(n_features, Y.shape[-1]//downsampling_entity, downsampling_entity).max(axis=2)
                    labels = labels.reshape(labels.shape[0]//downsampling_entity, downsampling_entity).max(axis=1)
                
                labels = labels[None, :]
                entity = Entity(Y=Y.reshape((1, -1)), name=meta_data['name'], labels=labels, verbose=verbose)
                entities.append(entity)

    data = Dataset(entities=entities, name=name, verbose=verbose)

    return data

def load_synthetic():
    pass

def load_swat(**kwargs):
    pass



