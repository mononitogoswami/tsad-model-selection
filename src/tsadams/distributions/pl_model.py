#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


class PlackettLuce(object):

    def __init__(self, scores: np.array, random_sate=0):
        self.scores = torch.from_numpy(np.array(scores))
        self.Z = torch.sum(self.scores)
        self.k = len(scores)
        torch.manual_seed(random_sate)

    def log_prob(self, p):
        """Find the log probability of a permutation. 

        TODO: Implement batching
        """
        logpmf = 0
        for i in range(self.k):
            Z_i = 0
            if i > 0:
                for j in range(i - 1):
                    Z_i = Z_i + self.scores[p[j]]
            logpmf += (torch.log(self.scores[p[i]]) - torch.log(self.Z - Z_i))
        return logpmf

    def prob(self, p):
        return torch.exp(self.log_prob(p))

    def sample(self, size=1):
        """Sample from a distribution
        """
        N_i = self.Z * torch.ones(size, 1)
        scores = self.scores.clone().reshape(1, self.k).tile((size, 1))
        p = torch.zeros((size, self.k), dtype=torch.int64)
        for i in range(self.k):
            pvals = scores / N_i
            pvals = pvals / torch.sum(pvals, axis=1, keepdims=True)
            p[:, i] = torch.multinomial(input=pvals,
                                        num_samples=1,
                                        replacement=True).squeeze()
            N_i = N_i - scores.gather(1, p[:, i].view((-1, 1)))
            scores = scores.scatter_(dim=1,
                                     index=p[:, i].view((-1, 1)),
                                     value=0.)
        return p
