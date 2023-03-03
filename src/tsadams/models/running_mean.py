import torch as t
import torch.nn as nn
from .base_model import PyMADModel
from ..utils.utils import de_unfold
import pandas as pd
import numpy as np

class RunningMean(PyMADModel):
    def __init__(self, 
                 n_features=1, 
                 running_window_size=60, 
                 window_size=-1, 
                 window_step=-1, 
                 device=None):
        super(RunningMean, self).__init__(window_size, window_step, device)
        # self._VALID_MEAN_ALGORITHMS = ['simple', 'exp']

        self.n_features = n_features
        self.running_window_size = running_window_size
        self.training_type = 'direct'
        # self.alpha = alpha

    def fit(self, train_dataloader, eval_dataloader=None):
        pass
        
    def forward(self, input):
        n_batches, n_features, n_time = input['Y'].shape
        assert n_batches == 1, 'Currently accepts batch size of 1'
        Y = input['Y'].reshape((n_features, n_time)).T
        # NOTE: Currently not using mask

        Y = Y.detach().numpy()
        
        Y = pd.DataFrame(Y)
        Y_hat = Y.rolling(self.running_window_size).mean().values
        Y = Y.values
        Y_hat[:self.running_window_size-1, :] = Y[:self.running_window_size-1, :]

        Y = Y.T#[None, :, :]
        Y_hat = Y_hat.T#[ :, :]

        return t.from_numpy(Y), t.from_numpy(Y_hat), input['mask']

    def window_anomaly_score(self, input, return_detail: bool=False):

        # Forward
        Y, Y_hat, mask = self.forward(input=input)

        # Anomaly Score
        anomaly_score = (Y - Y_hat)**2
        
        if return_detail:
            return anomaly_score
        else:
            return t.mean(anomaly_score, dim=0)

    def final_anomaly_score(self, input, return_detail: bool=False):
        
        # Average anomaly score for each feature per timestamp
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)

        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = t.mean(anomaly_scores, dim=0)
            return anomaly_scores