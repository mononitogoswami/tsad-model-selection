import torch as t
import torch.nn as nn
from .base_model import PyMADModel
from ..utils.utils import de_unfold


class _MeanDeviation(nn.Module):
    def __init__(self, n_features, device):
        super(_MeanDeviation, self).__init__()
        self.means = t.nn.Parameter(t.zeros((n_features, 1), device=device))

    def forward(self, input):
        x_hat = self.means*t.ones(input.shape, device=input.device)
        return x_hat

class MeanDeviation(PyMADModel):
    def __init__(self, n_features, window_size=1, window_step=1, device=None):
        super(MeanDeviation, self).__init__(window_size, window_step, device)

        self.n_features = n_features
        
        # Learnable parameters should be in self.model 
        self.model = _MeanDeviation(n_features=self.n_features, device=self.device)

        self.training_type = 'sgd'

    def forward(self, input):
        Y = input['Y'].to(self.device)
        mask = input['mask'].to(self.device)

        # Hide with mask
        Y = Y * mask

        Y_hat = self.model(Y)

        return Y, Y_hat, mask

    def training_step(self, input):
        self.model.train()

        Y, Y_hat, mask = self.forward(input=input)
        
        loss = t.mean((mask*(Y - Y_hat))**2)

        return loss

    def eval_step(self, x):
        self.model.eval()
        loss = self.training_step(x)
        return loss

    def window_anomaly_score(self, input, return_detail: bool=False):

        # Forward
        Y, Y_hat, mask = self.forward(input=input)

        # Anomaly Score
        anomaly_score = (mask*(Y - Y_hat))**2
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