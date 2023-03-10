#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages

setup(
    name="tsadams",
    version="0.1",
    description="Unsupervised Model Selection of Time-series Anomaly Detection Models",
    author="Mononito Goswami",
    author_email="mgoswami@andrew.cmu.edu",
    license="Apache v2.0",
    url="https://github.com/mononitogoswami/tsad-model-selection",
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        "cvxopt==1.3.0",
        "cvxpy==1.2.2",
        "matplotlib==3.6.2",
        "numpy==1.23.4",
        "networkx==2.8.8",
        "pandas==1.5.2",
        "patool==1.12",
        "scikit-learn==1.1.3",
        "scipy==1.9.3",
        "setuptools==65.5.0"
        "statsmodels==0.13.5",
        "tqdm==4.64.1"
        ]
)
