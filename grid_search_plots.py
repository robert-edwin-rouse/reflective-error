#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:37:02 2024

@author: robertrouse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import matplotlib as mpl
from apollo import mechanics as ma

### Set plotting style parameters
ma.textstyle()

### Load results data
avon = pd.read_csv('results/53018_grid_opt.csv')
exep = pd.read_csv('results/45009_grid_opt.csv')

### Set alpha and colourmap values
alphas = avon['Alpha'].unique()
colours = mpl.colormaps['viridis'](np.linspace(0, 1, len(alphas)))

### Define plotting function
def grid_plot(alphas, dataset, metric, y_label, y_limits):
    fig, ax = plt.subplots(figsize=(8, 8))
    for a in range(len(alphas)):
        subset = dataset[dataset['Alpha']==alphas[a]]
        plt.plot(subset['Delta'], subset[metric], c=colours[a], lw=3.2,
                 label=r'$\alpha$'+' = ' + str(alphas[a]))
        ax.set_xlabel(r'$\beta$'+ ' - ' + r'$\alpha$')
        ax.set_ylabel(y_label)
        ax.set_xlim([0, 10])
        ax.set_ylim(y_limits)
        ax.set_xscale('log')
        ax.yaxis.set_major_locator(mtk.MaxNLocator(5))
        ax.grid(lw=0.5, alpha=0.5)
    plt.legend(ncol=2, loc='upper left', labelspacing = 0.25)
    plt.show()

### Plot values for R2 and RE for alpha and beta pairs for each catchment
grid_plot(alphas, avon, 'R2', 'R' + r'$^2$', [0.82, 0.90])
grid_plot(alphas, avon, 'RE', 'RE', [0.82, 0.90])
grid_plot(alphas, exep, 'R2', 'R' + r'$^2$', [0.60, 0.85])
grid_plot(alphas, exep, 'RE', 'RE', [0.875, 1.00])