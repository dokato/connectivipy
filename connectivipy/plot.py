# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from six.moves import range

# plain plotting from values
def plot_conn(values, name='', fs=1, ylim=None, xlim=None, show=True):
    '''
    Plot connectivity estimation results. Allows to plot your results
    without using *Data* class.
    
    Args:
      *values* : numpy.array
       connectivity estimation values in shape (fq, k, k) where fq -
       frequency, k - number of channels 
      *name* = '' : str
        title of the plot
      *fs* = 1 : int
        sampling frequency
      *ylim* = None : list
        range of y-axis values shown, e.g. [0,1]
        *None* means that default values of given estimator are taken
        into account
      *xlim* = None : list [from (int), to (int)]
        range of y-axis values shown, if None it is from 0 to Nyquist frequency
      *show* = True : boolean
        show the plot or not            
    '''
    fq, k, k = values.shape
    fig, axes = plt.subplots(k, k)
    freqs = np.linspace(0, fs//2, fq)
    if not xlim:
        xlim = [0, np.max(freqs)]
    if not ylim:
            ylim = [np.min(values), np.max(values)]
    for i in range(k):
        for j in range(k):
            axes[i, j].fill_between(freqs, values[:, i, j], 0)
            axes[i, j].set_xlim(xlim)
            axes[i, j].set_ylim(ylim)
    plt.suptitle(name, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if show:
        plt.show()
