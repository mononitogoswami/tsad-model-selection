import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .base_model import PyMADModel
from ..utils.utils import de_unfold

class AnomalyAttention(nn.Module):
    def __init__(self, N, d, d_model, device):
        super(AnomalyAttention, self).__init__()
        self.d_model = d_model
        self.N = N
        self.device = device

        self.Wq = nn.Linear(d, d_model, bias=False)
        self.Wk = nn.Linear(d, d_model, bias=False)
        self.Wv = nn.Linear(d, d_model, bias=False)
        self.Ws = nn.Linear(d, 1, bias=False)

        self.Q = self.K = self.V = self.sigma = torch.zeros((N, d_model))

        self.P = torch.zeros((N, N))
        self.S = torch.zeros((N, N))

    def forward(self, x):

        self.initialize(x)
        self.P = self.prior_association()
        self.S = self.series_association()
        Z = self.reconstruction()
        return Z

    def initialize(self, x):
        self.Q = self.Wq(x)
        self.K = self.Wk(x)
        self.V = self.Wv(x)
        self.sigma = self.Ws(x)

    @staticmethod
    def gaussian_kernel(mean, sigma):
        normalize = 1 / (math.sqrt(2 * torch.pi) * torch.abs(sigma))
        return normalize.to(mean.device) * torch.exp(-0.5 * (mean / sigma).pow(2))

    def prior_association(self):
        p = torch.from_numpy(
            np.abs(np.indices((self.N, self.N))[0] - np.indices((self.N, self.N))[1])
        )
        p = p.float().to(self.device)
        gaussian = self.gaussian_kernel(p, self.sigma)
        #gaussian /= gaussian.sum(dim=-1).view(-1, 1)
        gaussian /= gaussian.sum(dim=-1).unsqueeze(2)

        return gaussian

    def series_association(self):
        # return F.softmax((self.Q @ self.K.T) / math.sqrt(self.d_model), dim=0)
        qk = torch.matmul(self.Q, self.K.transpose(1,2))
        return F.softmax( qk / math.sqrt(self.d_model), dim=-1)

    def reconstruction(self):
        return self.S @ self.V


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, N, d, d_model, device):
        super().__init__()
        self.N, self.d, self.d_model = N, d, d_model
        
        self.device = device
        self.attention = AnomalyAttention(self.N, self.d, self.d_model, device=device)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x_identity = x
        x = self.attention(x)
        if self.d!=self.d_model:
            z = self.ln1(x)
        else:
            z = self.ln1(x + x_identity)

        z_identity = z
        z = self.ff(z)
        z = self.ln2(z + z_identity)

        return z

class _AnomalyTransformer(nn.Module):
    def __init__(self, N, d, d_model, layers, device ):
        super().__init__()

        self.N = N
        self.d_model = d_model
        self.d = d
        self.device = device

        self.w_rec = nn.Linear(d_model, d)

        self.blocks = [AnomalyTransformerBlock(self.N, self.d, self.d_model, device=self.device)]        
        self.blocks.extend([AnomalyTransformerBlock(self.N, self.d_model, self.d_model, device=self.device) for _ in range(1, layers)])
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        return self.blocks(x)

class AnomalyTransformer(PyMADModel):
    def __init__(self, window_size, window_step, d, d_model, layers, lambda_, device=None):
        super(AnomalyTransformer, self).__init__(window_size, window_step, device)

        self.N, self.window_size = window_size, window_size
        self.window_step = window_step
        self.d_model = d_model
        self.d = d
        self.layers= layers
        self.training_type = 'sgd'
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.output = None
        self.lambda_ = lambda_

        self.model = _AnomalyTransformer(N=window_size, d=d, d_model=d_model, layers=layers, device=self.device).to(self.device)

    def training_step(self, input):

        Y = input['Y'].clone().to(self.device) #TODO: revisar si clone es necesario
        mask = input['mask'].clone().to(self.device)
        Y = Y.permute(0,2,1)
        mask = mask.permute(0,2,1)

        batch_idx = input['idx']

        self.model.train()

        outputs = self(Y)
        min_loss = self.min_loss(Y)
        max_loss = self.max_loss(Y)

        return min_loss, max_loss

    def eval_step(self, x):
        assert 1<0, 'NOT IMPLEMENTED YET'
        self.model.eval()
        min_loss, max_loss = self.training_step(x)
        return min_loss, max_loss

    def forward(self, x):
        self.P_layers = []
        self.S_layers = []
        for idx, block in enumerate(self.model.blocks):
            x = block(x)
            self.P_layers.append(block.attention.P)
            self.S_layers.append(block.attention.S)

        self.output = self.model.w_rec(x)
        return x

    def layer_association_discrepancy(self, Pl, Sl, x):
        rowwise_kl = lambda row: (
            F.kl_div(Pl[row, :], Sl[row, :]) + F.kl_div(Sl[row, :], Pl[row, :])
        )
        ad_vector = torch.concat(
            [rowwise_kl(row).unsqueeze(0) for row in range(Pl.shape[0])]
        )
        return ad_vector

    def layer_association_discrepancy_batch(self, Pl, Sl, x):
        ad_vector = torch.concat(
            [self.rowwise_kl(row, Pl, Sl).unsqueeze(1) for row in range(Pl.shape[1])], dim=1
        )
        return ad_vector

    def rowwise_kl(self, row, Pl, Sl, eps=1e-4):
        Pl_r = (Pl[:, row, :] + eps) / torch.sum(Pl[:, row, :] + eps, dim=-1, keepdims=True)
        Sl_r = (Sl[:, row, :] + eps) / torch.sum(Sl[:, row, :] + eps, dim=-1, keepdims=True)
        return torch.sum( 
            F.kl_div( torch.log(Pl_r), Sl_r, reduction='none') + F.kl_div( torch.log(Sl_r), Pl_r, reduction='none'), dim=1
         )

    def association_discrepancy(self, P_list, S_list, x):
        return (1 / len(P_list)) * sum(
            [
                self.layer_association_discrepancy_batch(P, S, x)
                for P, S in zip(P_list, S_list)
            ]
        )

    def loss_function(self, x_hat, P_list, S_list, lambda_, x):
        frob_norm = torch.pow(torch.linalg.matrix_norm(x_hat - x, ord="fro"), 2) # squared or not
        # frob_norm = torch.linalg.matrix_norm(x_hat - x, ord="fro")
        return torch.mean(
                frob_norm - (
                lambda_
                * torch.linalg.norm(self.association_discrepancy(P_list, S_list, x), ord=1, dim=1)
            )
        ) / self.N

    def loss_function_log(self, x):
        P_list = [P.detach() for P in self.P_layers]
        S_list = [S.detach() for S in self.S_layers]
        frob_norm = torch.mean( torch.pow(torch.linalg.matrix_norm(self.output - x, ord="fro"), 2) )  / self.N
        ass_dis = torch.mean( torch.linalg.norm(self.association_discrepancy(P_list, S_list, x), ord=1, dim=1) ) / self.N
        return frob_norm, ass_dis

    def min_loss(self, x):
        P_list = self.P_layers
        S_list = [S.detach() for S in self.S_layers]
        lambda_ = -self.lambda_
        return self.loss_function(self.output, P_list, S_list, lambda_, x)

    def max_loss(self, x):
        P_list = [P.detach() for P in self.P_layers]
        S_list = self.S_layers
        lambda_ = self.lambda_
        return self.loss_function(self.output, P_list, S_list, lambda_, x)

    def window_anomaly_score(self, input, return_detail: bool=False):
        Y = input['Y'].clone().to(self.device) #TODO: revisar si clone es necesario
        mask = input['mask'].clone().to(self.device)
        batch_size, n_features, _ = Y.shape

        Y = Y.permute(0,2,1)

        # Forward
        self(Y)

        ad = F.softmax(
            -self.association_discrepancy(self.P_layers, self.S_layers, Y), dim=1
        )
        assert ad.shape[1] == self.N
        norm = torch.concat(
            [
                torch.pow(torch.linalg.norm(Y[:, i, :] - self.output[:, i, :], ord=2, dim=1).unsqueeze(1) , 2)
                for i in range(self.N)
            ], dim=1
        )
        assert norm.shape[1] == self.N
        score = torch.mul(ad, norm)

        # Broadcast scores to n_features and window_size
        if return_detail:
            scores = torch.ones((batch_size, n_features, self.window_size))*score[:, None, :]
            return scores

    def final_anomaly_score(self, input, return_detail: bool=False):
        
        # Average anomaly score for each feature per timestamp
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)

        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = torch.mean(anomaly_scores, dim=0)
            return anomaly_scores
