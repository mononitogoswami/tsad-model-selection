#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import setuptools

module_name = input("Enter module name: ")
setuptools.setup(
    name=module_name,
    py_modules=[module_name],
)
