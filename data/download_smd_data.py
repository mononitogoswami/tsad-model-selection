#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to download the whole SMD data
#######################################

import sys

sys.path.append('/home/ubuntu/PyMAD/')  # TODO: Make this relative path maybe
from src.pymad.datasets.load import load_data

_ = load_data(dataset='smd',
              group='train',
              entities=None,
              downsampling=None,
              min_length=None,
              root_dir='/home/ubuntu/datasets/',
              verbose=False)
