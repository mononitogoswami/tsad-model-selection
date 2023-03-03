import numpy as np
import torch as t

class PyMADModel(t.nn.Module):
    def __init__(self, window_size=1, window_step=1, device=None):
        super(PyMADModel, self).__init__()

        self.window_size = window_size
        self.window_step = window_step

        # State Flags
        self.is_trained = False
        self.training = False
        self.predicting = False

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def forward(self, input):
        pass

    def training_step(self, input):
        pass

    def eval_step(self, input):
        pass

    def window_anomaly_score(self, input, return_detail: bool=False):
        pass

    def final_anomaly_score(self, input, return_detail: bool=False):
        pass

    def __str__(self):
        string = str({
            'window_size': self.window_size,
            'window_step': self.window_step
        })
        return string