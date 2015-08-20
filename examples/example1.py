# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np 
import connectivipy as cp
from connectivipy import mvar_gen

### MVAR model coefficients

# let's build mvar model matrix
A = np.zeros((2, 5, 5))
# 2 - first dimension is model order
# 5 - second and third dimensions mean number of channels
A[0, 0, 0] = 0.95 * 2**0.5
A[1, 0, 0] = -0.9025
A[0, 1, 0] = -0.5
A[1, 2, 1] = 0.4
A[0, 3, 2] = -0.5
A[0, 3, 3] = 0.25 * 2**0.5
A[0, 3, 4] = 0.25 * 2**0.5
A[0, 4, 3] = -0.25 * 2**0.5
A[0, 4, 4] = 0.25 * 2**0.5

# multitrial signal generation from a matrix above
# let's generate 5-channel signal with 1000 data points
# and 5 trials using function mvar_gen
ysig = np.zeros((5, 10**3, 5))
ysig[:,:,0] = mvar_gen(A,10**3)
ysig[:,:,1] = mvar_gen(A,10**3)
ysig[:,:,2] = mvar_gen(A,10**3)
ysig[:,:,3] = mvar_gen(A,10**3)
ysig[:,:,4] = mvar_gen(A,10**3)

#### connectivity analysis

data = cp.Data(ysig,128, ["Fp1", "Fp2","Cz", "O1","O2"])

# you may want to plot data (in multitrial case only one trial is shown)
data.plot_data()

# fit mvar using Yule-Walker algorithm and order 2
data.fit_mvar(2,'yw')

# you can capture fitted parameters and residual matrix
ar, vr = data.mvar_coefficients 

# now we investigate connectivity using gDTF
gdtf_values = data.conn('gdtf')
gdtf_significance = data.significance(Nrep=200, alpha=0.05)
data.plot_conn('gDTF')

# short time version with default parameters
pdc_shorttime = data.short_time_conn('pdc', nfft=100, no=10)
data.plot_short_time_conn("PDC")
