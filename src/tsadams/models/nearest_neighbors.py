import torch as t
from sklearn.neighbors import NearestNeighbors as NN
from .base_model import PyMADModel
from ..utils.utils import de_unfold

class NearestNeighbors(PyMADModel):
    def __init__(self, n_neighbors, window_size=1, window_step=1):
        
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.window_step = window_step

        self.model = NN(n_neighbors=n_neighbors, algorithm='ball_tree')
        # TODO: Can also make the distance metric

        self.training_type = 'direct'
        self.device = None

    def fit(self, train_dataloader, eval_dataloader=None):
        
        Y_windows = train_dataloader.Y_windows.reshape(len(train_dataloader.Y_windows), -1)
        # TODO: One can do PCA for dimensionality reduction

        self.model.fit(X=Y_windows)

    def forward(self, input):
        Y_windows = input['Y']
        batch_size, n_features, _ = Y_windows.shape

        # Flatten window
        Y_windows =  Y_windows.reshape(batch_size, -1)

        # Compute distances to train windows
        _, indices = self.model.kneighbors(Y_windows)
        
        Y_hat = self.model._fit_X[indices].mean(axis=1).reshape((batch_size, n_features, -1))
        return input['Y'], t.from_numpy(Y_hat), input['mask']

    def window_anomaly_score(self, input, return_detail):

        Y_windows = input['Y']
        batch_size, n_features, _ = Y_windows.shape

        # Flatten window
        Y_windows =  Y_windows.reshape(batch_size, -1)

        # Compute distances to train windows
        distances, _ = self.model.kneighbors(Y_windows)
        scores = distances.mean(axis=1)

        # Broadcast scores to n_features and window_size
        if return_detail:
            scores = t.ones((batch_size, n_features, self.window_size))*scores[:, None, None]

        return scores

    def final_anomaly_score(self, input, return_detail: bool=False):
        
        # Average anomaly score for each feature per timestamp
        anomaly_scores = de_unfold(windows=t.Tensor(input), window_step=self.window_step)

        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = t.mean(anomaly_scores, dim=0)
            return anomaly_scores 

    def eval(self):
        pass