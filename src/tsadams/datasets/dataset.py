
import numpy as np
import torch as t

from typing import List, Optional

#TODO: think how to group train-test
#TODO: prints with tabs
#TODO: downsampling
#TODO: rename dataset object
#TODO: ds

class Entity(object):
    def __init__(self,
                 Y: np.ndarray,
                 X: Optional[np.ndarray] = None,
                 name: Optional[str] = None,
                 labels: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None,
                 verbose: bool = False) -> 'Entity':
        """
        Parameters
        ----------
        Y: np.ndarray (m, t)
            Multivariate time-series data, m is the number of features, t is the number of timestamps.
        X: np.ndarray (m2, t)
            Time-series exogenous covariates, m2 is the number of exogenous features, t is the number of timestamps.
        name: str
            Name of the entity.
        labels: np.ndarray (1, t)
            Binary label for anomalies. 1: anomaly, 0: non-anomaly. If not passed, assumes all is non-anomaly.
        mask: np.ndarray (m, t)
            Indicator to mask/hide features on each timestamp. 1: use point, 0: hide point. If not passed, assumes use all points.
        verbose:
            Boolean for printing details.
        """
        # Data
        self.Y = Y
        self.n_features, self.n_time = Y.shape



        # Exogenous Covariates
        if X is None:
            self.X = None
            self.n_exogenous = 0
        else:
            n_exogenous, n_x_time = X.shape
            assert n_x_time == self.n_time, 'Exogenous covariates and Y should have same number of timestamps.'
            self.X = X
            self.n_exogenous = n_exogenous

        # Labels
        if labels is None:
            self.labels = np.zeros((1, self.n_time))
        else:
            assert labels.shape[-1] == self.n_time, 'Data and labels should have same number of timestamps.'
            self.labels = labels

        # Name
        if name is None:
            self.name = 'XXX' #TODO: random generator
        else:
            self.name = name

        # Mask
        if mask is None:
            self.mask = np.ones((self.n_features, self.n_time))
        else:
            assert mask.shape == Y.shape, 'Y and mask should have same shape.'
            self.mask = mask

        self.verbose = verbose
        if self.verbose:
            print(42*'-')
            print('Entity: ', self.name)
            print('n_features: ', self.n_features)
            print('n_exogenous: ', self.n_exogenous)
            print('n_time: ', self.n_time)
            print('Anomaly %: ', np.mean(self.labels))
            print('Mask %: ', np.mean(self.mask))
            print(42*'-')


class Dataset(object):
    def __init__(self,
                 entities: List[Entity],
                 name: str,
                 verbose: bool = False) -> 'Dataset':
        """
        Parameters
        ----------
        entities: List[np.ndarray]
            List of Entities objects
        name: str
            Name of the dataset.
        verbose:
            Boolean for printing details.
        """

        self.entities = entities
        self.name = name
        self.verbose = verbose

        # Important information
        self.n_features = entities[0].n_features
        self.n_exogenous = entities[0].n_exogenous
        self.n_entities = len(entities)

        self.total_time = 0
        self.n_anomalies = 0
        for entity in entities:
            self.total_time += entity.n_time
            self.n_anomalies += np.sum(entity.labels)
        
        # asserts
        assert all([(entity.n_features == self.n_features) for entity in self.entities]), 'All entities must have same number of features!'
        assert all([(entity.n_exogenous == self.n_exogenous) for entity in self.entities]), 'All entities must have same number of exogenous variables!'

        if self.verbose:
            print(42*'-')
            print(self)
            print(42*'-')
                        
    def get_entities(self, entities_names: List[str], dataset_name):
        entities = []
        for entity in self.entities:
            if entity.name in entities_names:
                entities.append(entity)
        return Dataset(entities=entities, name=dataset_name)

    def get_entity(self, entity_name: str):
        """
        Returns a specific entity in the dataset
        """
        for entity in self.entities:
            if entity.name == entity_name:
                return entity
            else: 
                raise ValueError('Entity not found!')

    def __len__(self):
        return self.n_entities

    def __iter__(self):
        for entity in self.entities:
            yield entity

    def __str__(self):
        info_dict = {'Dataset': self.name,
                     'n_entities': self.n_entities,
                     'n_features': self.n_features,
                     'n_exogenous': self.n_exogenous,
                     'total_time': self.total_time,
                     'anomaly_percentage': 100*(self.n_anomalies/self.total_time)}

        info_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in info_dict.items()}
        attrs_as_str = [f"\t{k}={v},\n" for k, v in info_dict.items()]

        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"