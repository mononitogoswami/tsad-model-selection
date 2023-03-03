"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import PyMADModel
from ..utils.utils import de_unfold


class Generator(nn.Module):
    def __init__(self, window_size=32, hidden_multiplier=32, latent_size=100, n_features=3, max_filters=256, kernel_multiplier=1):
        super(Generator, self).__init__()

        n_layers = int(np.log2(window_size))
        layers = []
        filters_list = []
        # First layer
        filters = min(max_filters, hidden_multiplier*(2**(n_layers-2)))
        layers.append(nn.ConvTranspose1d(in_channels=latent_size, out_channels=filters,
                                         kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(filters))
        filters_list.append(filters)
        # Hidden layers
        for i in reversed(range(1, n_layers-1)):
            filters = min(max_filters, hidden_multiplier*(2**(i-1)))
            layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=filters,
                                             kernel_size=4*kernel_multiplier, stride=2, padding=1 + (kernel_multiplier-1)*2, bias=False))
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())
            filters_list.append(filters)

        # Output layer
        layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=n_features, kernel_size=3, stride=1, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, input, mask=None):
        input = input[:,:,0,:]
        input = self.layers(input)
        input = input[:,:,None,:]
        
        # Hide mask
        if mask is not None:
            input = input * mask

        return input

class DGHL(PyMADModel):
    def __init__(self, window_size, window_step, a_L,
                 n_features, hidden_multiplier, max_filters, kernel_multiplier,
                 z_size, z_size_up, z_iters, z_iters_inference, z_sigma, z_step_size, z_with_noise, z_persistent,
                 noise_std, normalize_windows, random_seed, device=None):
        super(DGHL, self).__init__(window_size, window_step, device)

        assert z_persistent == False

        # Generator
        self.n_features = n_features
        self.hidden_multiplier = hidden_multiplier
        self.z_size = z_size
        self.z_size_up = z_size_up
        self.max_filters = max_filters
        self.kernel_multiplier = kernel_multiplier
        self.normalize_windows = normalize_windows
        self.a_L = a_L

        # Alternating back-propagation
        self.z_iters = z_iters
        self.z_iters_inference = z_iters_inference
        self.z_sigma = z_sigma
        self.z_step_size = z_step_size
        self.z_with_noise = z_with_noise
        self.z_persistent = z_persistent
        self.p_0_chains_u = None
        self.p_0_chains_l = None

        # Training
        self.noise_std = noise_std

        # Generator
        torch.manual_seed(random_seed)

        self.sub_window_size = int(self.window_size/a_L)

        # Learnable parameters should be in self.model
        self.model = Generator(window_size=self.sub_window_size, hidden_multiplier=self.hidden_multiplier,
                               latent_size=self.z_size+self.z_size_up,
                               n_features=self.n_features, max_filters=self.max_filters,
                               kernel_multiplier=self.kernel_multiplier).to(self.device)

        self.training_type = 'sgd'

        self.mse_loss = nn.MSELoss(reduction='sum')

    def infer_z(self, z, Y, mask, n_iters, with_noise):
        
        z_u = z[0]
        z_l = z[1]

        for i in range(n_iters):
            z_u = torch.autograd.Variable(z_u, requires_grad=True)
            z_l = torch.autograd.Variable(z_l, requires_grad=True)
            
            z_u_repeated = torch.repeat_interleave(z_u, self.a_L, 0)
            z = torch.cat((z_u_repeated, z_l), dim=1).to(self.device)

            Y_hat = self.model(z, mask)

            L = 1.0 / (2.0 * self.z_sigma * self.z_sigma) * self.mse_loss(Y_hat, Y)
            L.backward()
            z_u = z_u - 0.5 * self.z_step_size * self.z_step_size * (z_u + z_u.grad)
            z_l = z_l - 0.5 * self.z_step_size * self.z_step_size * (z_l + z_l.grad)
            if with_noise:
                eps_u = torch.randn(len(z_u), self.z_size_up, 1, 1).to(z_u.device)
                z_u += self.z_step_size * eps_u
                eps_l = torch.randn(len(z_l), self.z_size, 1, 1).to(z_l.device)
                z_l += self.z_step_size * eps_l

        z_u = z_u.detach()
        z_l = z_l.detach()
        z = z.detach()

        return z, z_u, z_l

    def sample_gaussian(self, n_dim, n_samples):
        p_0 = torch.distributions.MultivariateNormal(torch.zeros(n_dim), 0.01*torch.eye(n_dim))
        p_0 = p_0.sample([n_samples]).view([n_samples, -1, 1, 1])

        return p_0

    def _preprocess_batch_hierarchy(self, windows):
        """
        X tensor of shape (batch_size, n_features, 1, window_size*a_L)
        """
        batch_size, n_features, _, window_size = windows.shape

        assert n_features == self.n_features, f'Batch n_features {n_features} not consistent with Generator'
        assert window_size == self.window_size, f'Window size {window_size} not consistent with Generator'

        # Wrangling from (batch_size, n_features, 1, window_size*window_hierarchy) -> (batch_size*window_hierarchy, n_features, 1, window_size)
        windows = windows.unfold(dimension=-1, size=self.sub_window_size, step=self.sub_window_size)
        windows = windows.swapaxes(1,3)
        windows = windows.swapaxes(2,3)
        windows = windows.reshape(batch_size*self.a_L, self.n_features, 1, self.sub_window_size)

        return windows

    def _postprocess_batch_hierarchy(self, windows):

        # Return to window_size * window_hierarchy size
        windows = windows.swapaxes(0,2)
        windows = windows.reshape(1,self.n_features, -1, self.window_size)
        windows = windows.swapaxes(0,2)

        return windows

    def _get_initial_z(self, p_0_chains_u, p_0_chains_l, z_persistent, idx):
        
        batch_size = len(idx)

        # if z_persistent:
        #     p_0_z_u = p_0_chains_u[i]
        #     p_0_z_l = p_0_chains_l[i]
        #     p_0_z_u = p_0_z_u.reshape(batch_size, self.z_size_up,1,1)
        #     p_0_z_l = p_0_z_l.reshape(batch_size*self.window_hierarchy, self.z_size,1,1)
        # else:
        #     p_0_z_u = self.sample_gaussian(n_dim=self.z_size_up, n_samples=batch_size)
        #     p_0_z_u = p_0_z_u.to(self.device)

        #     p_0_z_l = self.sample_gaussian(n_dim=self.z_size, n_samples=batch_size*self.window_hierarchy)
        #     p_0_z_l = p_0_z_l.to(self.device)

        p_0_z_u = self.sample_gaussian(n_dim=self.z_size_up, n_samples=batch_size)
        p_0_z_u = p_0_z_u.to(self.device)

        p_0_z_l = self.sample_gaussian(n_dim=self.z_size, n_samples=batch_size*self.a_L)
        p_0_z_l = p_0_z_l.to(self.device)

        p_0_z = [p_0_z_u, p_0_z_l]

        return p_0_z

    def forward(self, input):

        Y = input['Y'].clone().to(self.device) #TODO: revisar si clone es necesario
        mask = input['mask'].clone().to(self.device)
        batch_idx = input['idx']

        Y = Y[:, :, None, :] # Legacy code 
        mask = mask[:, :, None, :] # Legacy code

        # Append exogenous data
        if len(input['X']) > 0:
            X = input['X'].clone().to(self.device) #TODO: revisar si clone es necesario
            X = X[:, :, None, :] # Legacy code
            Y = torch.cat([Y, X], dim=1)
            mask = torch.tile(mask, dims=[1, Y.shape[1], 1, 1])

        # Normalize windows # TODO: be careful, mask not considered yet
        x_scales = Y[:,:,:,[0]]
        if self.normalize_windows:
            Y = Y - x_scales
            x_scales = x_scales.to(self.device)

        # Hide with mask
        Y = Y * mask
        
        # Gaussian noise, not used if generator is in eval mode
        if self.model.train:
            Y = Y + self.noise_std*(torch.randn(Y.shape).to(self.device))

        Y = self._preprocess_batch_hierarchy(windows=Y)
        mask = self._preprocess_batch_hierarchy(windows=mask)

        z_0 = self._get_initial_z(p_0_chains_l=self.p_0_chains_l, p_0_chains_u=self.p_0_chains_u,
                                  z_persistent=self.z_persistent, idx=batch_idx)

        # Sample z with Langevin Dynamics
        z, z_u, z_l = self.infer_z(z=z_0, Y=Y, mask=mask, n_iters=self.z_iters, with_noise=self.z_with_noise)
        Y_hat = self.model(input=z, mask=mask)

        Y = self._postprocess_batch_hierarchy(windows=Y)
        mask = self._postprocess_batch_hierarchy(windows=mask)
        Y_hat = self._postprocess_batch_hierarchy(windows=Y_hat)

        if self.normalize_windows:
            Y = Y + x_scales
            Y_hat = Y_hat + x_scales
            Y = Y * mask
            Y_hat = Y_hat * mask
        
        Y = Y.squeeze(2)
        Y_hat = Y_hat.squeeze(2)
        mask = mask.squeeze(2)

        return Y, Y_hat, mask

    def compute_loss(self, Y, Y_hat):
        # Loss
        loss = 0.5 * self.mse_loss(Y, Y_hat)
        return loss

    def training_step(self, input):
        # TODO: Persistent
        # if self.z_persistent:
        #     self.p_0_chains_u = torch.zeros((X.shape[0],1,self.z_size_up,1,1))
        #     self.p_0_chains_l = torch.zeros((X.shape[0], self.window_hierarchy, self.z_size,1,1))
        #     for i in range(X.shape[0]):
        #         p_0_chains_u = self.sample_gaussian(n_dim=self.z_size_up, n_samples=1)
        #         p_0_chains_u = p_0_chains_u.to(self.device)
        #         self.p_0_chains_u[i] = p_0_chains_u

        #         p_0_chains_l = self.sample_gaussian(n_dim=self.z_size, n_samples=self.window_hierarchy)
        #         p_0_chains_l = p_0_chains_l.to(self.device)
        #         self.p_0_chains_l[i] = p_0_chains_l

        self.model.train()

        Y, Y_hat, mask = self.forward(input)
        loss = self.compute_loss(Y, Y_hat)

        return loss
    
    def eval_step(self, x):
        self.model.eval()
        loss = self.training_step(x)
        return loss

    def window_anomaly_score(self, input, return_detail: bool=False):

        self.model.eval()

        Y = input['Y'].clone().to(self.device) #TODO: revisar si clone es necesario
        mask = input['mask'].clone().to(self.device)
        batch_idx = input['idx']

        n_features = Y.shape[1]

        Y = Y[:, :, None, :] # Legacy code 
        mask = mask[:, :, None, :] # Legacy code
        # Append exogenous data
        if len(input['X']) > 0:
            X = input['X'].clone().to(self.device) #TODO: revisar si clone es necesario
            X = X[:, :, None, :] # Legacy code
            Y = torch.cat([Y, X], dim=1)
            mask = torch.tile(mask, dims=[1, Y.shape[1], 1, 1])

        # Normalize windows # TODO: be careful, mask not considered yet
        x_scales = Y[:,:,:,[0]]
        if self.normalize_windows:
            Y = Y - x_scales
            x_scales = x_scales.to(self.device)

        # Hide with mask
        Y = Y * mask

        Y = self._preprocess_batch_hierarchy(windows=Y)
        mask = self._preprocess_batch_hierarchy(windows=mask)

        z_0 = self._get_initial_z(p_0_chains_l=self.p_0_chains_l, p_0_chains_u=self.p_0_chains_u,
                                  z_persistent=self.z_persistent, idx=batch_idx)

        # Sample z with Langevin Dynamics
        z, _, _ = self.infer_z(z=z_0, Y=Y, mask=mask, n_iters=self.z_iters_inference, with_noise=False)
        mask = torch.ones(mask.shape).to(self.device) # In forward of generator, mask is all ones to reconstruct everything
        Y_hat = self.model(input=z, mask=mask)

        Y = self._postprocess_batch_hierarchy(windows=Y)
        mask = self._postprocess_batch_hierarchy(windows=mask)
        Y_hat = self._postprocess_batch_hierarchy(windows=Y_hat)

        Y_flatten = Y.squeeze(2)
        Y_hat_flatten = Y_hat.squeeze(2)
        mask_flatten = mask.squeeze(2)

        # Filter exogenous variables from tensors
        if len(input['X']) > 0:
            Y_flatten = Y_flatten[:, :n_features, :]
            Y_hat_flatten = Y_hat_flatten[:, :n_features, :]
            mask_flatten = mask_flatten[:, :n_features, :]

        ts_score = (Y_flatten-Y_hat_flatten)**2

        if return_detail:
            return ts_score*mask_flatten
        else:
            return np.average(ts_score, axis=1, weights=mask_flatten)

    def final_anomaly_score(self, input, return_detail: bool=False):
        
        # Average anomaly score for each feature per timestamp
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)

        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = torch.mean(anomaly_scores, dim=0)
            return anomaly_scores

    # def anomaly_score(self, X, mask, z_iters):
    #     x, x_hat, z, mask = self.predict(X=X, mask=mask, z_iters=z_iters)
    #     x_hat = x_hat*mask # Hide non-available data

    #     x_flatten = x.squeeze(2)
    #     x_hat_flatten = x_hat.squeeze(2)
    #     mask_flatten = mask.squeeze(2)
    #     z = z.squeeze((2,3))

    #     ts_score = np.square(x_flatten-x_hat_flatten)

    #     score = np.average(ts_score, axis=1, weights=mask_flatten)

    #     return score, ts_score, x, x_hat, z, mask