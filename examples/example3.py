import numpy as np
import matplotlib.pyplot as plt
import connectivipy as cp

"""
This example reproduce simulation from article:
Erla S et all (2009) "Multivariate autoregressive model with
                      instantaneous effects to improve brain
                      connectivity estimation"
"""

# let's make a matrix from original article

bcf = np.zeros((4, 5, 5))
# matrix shape meaning (k, k, p) k - number of channels,
# p - order of mvar parameters
bcf[1, 0, 0] = 1.58
bcf[2, 0, 0] = -0.81
bcf[0, 1, 0] = 0.9
bcf[2, 1, 1] = -0.01
bcf[3, 1, 4] = -0.6
bcf[1, 2, 1] = 0.3
bcf[1, 2, 2] = 0.8
bcf[2, 2, 1] = 0.3
bcf[2, 2, 2] = -0.25
bcf[3, 2, 1] = 0.3
bcf[0, 3, 1] = 0.9
bcf[1, 3, 1] = -0.6
bcf[3, 3, 1] = 0.3
bcf[1, 4, 3] = -0.3
bcf[2, 4, 0] = 0.9
bcf[2, 4, 3] = -0.3
bcf[3, 4, 2] = 0.6

# now we build a corresponding MVAR process without instantenous effect
L = np.linalg.inv(np.eye(5)-bcf[0])
acf = np.zeros((3, 5, 5))
for i in xrange(3):
    acf[i] = np.dot(L, bcf[i+1])

# generate 5-channel signals from matrix above
signal_inst = cp.mvar_gen_inst(bcf, int(10e4))
signal = cp.mvar_gen(acf, int(10e4))

# fit MVAR parameters
bv, vfb = cp.Mvar.fit(signal_inst, 3, 'yw')

av, vfa = cp.Mvar.fit(signal, 3, 'yw')

# use connectivity estimators
ipdc = cp.conn.iPDC()
ipdcval = ipdc.calculate(bv, vfb, 1.)

pdc = cp.conn.PDC()
pdcval = pdc.calculate(av, vfa, 1.)

def plot_double_conn(values_a, values_b, name='', fs=1, ylim=None, xlim=None, show=True):
    "function to plot two sets of connectivity values"
    fq, k, k = values_a.shape
    fig, axes = plt.subplots(k, k)
    freqs = np.linspace(0, fs*0.5, fq)
    if not xlim:
        xlim = [0, np.max(freqs)]
    if not ylim:
        ylim = [0, 1]
    for i in xrange(k):
        for j in xrange(k):
            axes[i, j].fill_between(freqs, values_b[:, i, j], 0, facecolor='red', alpha=0.5)
            axes[i, j].fill_between(freqs, values_a[:, i, j], 0, facecolor='black', alpha=0.5)
            axes[i, j].set_xlim(xlim)
            axes[i, j].set_ylim(ylim)
    plt.suptitle(name,y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if show:
        plt.show()

plot_double_conn(pdcval**2, ipdcval**2, 'PDC / iPDC')

