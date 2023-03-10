#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to download the Server Machine and Anomaly Archive datasets
#######################################

from tsadams.datasets.load import load_data
from tsadams.utils.utils import get_args_from_cmdline

def main():
    args = get_args_from_cmdline()

    _ = load_data(dataset='smd',
              group='train',
              entities=None,
              downsampling=None,
              min_length=None,
              root_dir=args['dataset_path'],
              verbose=False)

    _ = load_data(dataset='anomaly_archive',
                group='train',
                entities=None,
                downsampling=None,
                min_length=None,
                root_dir=args['dataset_path'],
                verbose=False)

if __name__ == '__main__':
    main()    