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
    zip_safe=False,
    packages=find_packages()
)
