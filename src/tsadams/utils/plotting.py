#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

plt.style.use('ggplot')


def box_plot_per_entity(aggregate_stats, dataset, split, metric):
    fig, axes = plt.subplots(1, 1, figsize=(4, 4), sharey=True, sharex=True)

    result_table = {}
    axes.set_title(
        f"Server Machine Dataset (n={len(aggregate_stats['Oracle No-MS/Oracle MS'])})",
        fontsize=10)
    _ = axes.boxplot([
        aggregate_stats['Oracle No-MS/Oracle MS'],
        aggregate_stats['Random MS/Oracle MS'],
        aggregate_stats['Trimmed Kemeny MS/Oracle MS'],
        aggregate_stats['Kemeny MS/Oracle MS'],
        aggregate_stats['Trimmed Borda MS/Oracle MS'],
        aggregate_stats['Borda MS/Oracle MS'],
        aggregate_stats['Forecasting-based MS/Oracle MS'],
        aggregate_stats['Centrality-based MS/Oracle MS'],
        aggregate_stats['Anomaly Injection-based MS/Oracle MS']
    ],
                     vert=False,
                     bootstrap=10000,
                     showmeans=True,
                     meanline=True,
                     labels=[
                         'Oracle No-MS', 'Random MS', 'Trimmed Kemeny MS',
                         'Kemeny MS', 'Trimmed Borda MS', 'Borda MS',
                         'Forecasting-based MS', 'Centrality-based MS',
                         'Anomaly Injection-based MS'
                     ])

    result_table['All Machines'] = {
        'Oracle No-MS':
        np.median(aggregate_stats['Oracle No-MS/Oracle MS']),
        'Random MS':
        np.median(aggregate_stats['Random MS/Oracle MS']),
        'Trimmed Kemeny MS':
        np.median(aggregate_stats['Trimmed Kemeny MS/Oracle MS']),
        'Kemeny MS':
        np.median(aggregate_stats['Kemeny MS/Oracle MS']),
        'Trimmed Borda MS':
        np.median(aggregate_stats['Trimmed Borda MS/Oracle MS']),
        'Borda MS':
        np.median(aggregate_stats['Borda MS/Oracle MS']),
        'Forecasting-based MS':
        np.median(aggregate_stats['Forecasting-based MS/Oracle MS']),
        'Centrality-based MS':
        np.median(aggregate_stats['Centrality-based MS/Oracle MS']),
        'Anomaly Injection-based MS':
        np.median(aggregate_stats['Anomaly Injection-based MS/Oracle MS']),
    }

    plt.savefig(
        f"box_plot_{dataset}_{split}_{metric}_{datetime.today().strftime(r'%H-%M-%m-%d-%Y')}.pdf",
        bbox_inches='tight')
    plt.show()
