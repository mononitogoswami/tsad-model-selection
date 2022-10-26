#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to download the whole SMD data
#######################################

import sys

sys.path.append('/zfsauton2/home/mgoswami/PyMAD')
from src.pymad.datasets.load import load_data

DATA_SAVE_DIR = r'/home/scratch/mgoswami/datasets/'
_ = load_data(dataset='smd',
              group='train',
              entities=None,
              downsampling=None,
              min_length=None,
              root_dir=DATA_SAVE_DIR,
              verbose=False)

_ = load_data(dataset='anomaly_archive',
              group='train',
              entities=None,
              downsampling=None,
              min_length=None,
              root_dir=DATA_SAVE_DIR,
              verbose=False)
