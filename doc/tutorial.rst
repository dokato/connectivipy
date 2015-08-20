.. _tutorial:

Tutorials
==================

Installation
########

The easiest way is to use *GIT* and just execute:

.. code-block:: bash

    $ git clone https://github.com/dokato/connectivipy.git
    $ cd connectivipy
    $ python setup.py install

Data class example
########

.. code-block:: python
    
    # Example 1

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


How to use specific classes
########

.. code-block:: python
    
    # Example 2

    import numpy as np
    import connectivipy as cp

    fs = 256.
    acf = np.zeros((3,3,3))
    # matrix shape meaning (p,k,k) k - number of channels,
    # p - order of mvar parameters

    acf[0,0,0] = 0.3
    acf[0,1,0] = 0.6
    acf[1,0,0] = 0.1
    acf[1,1,1] = 0.2
    acf[1,2,0] = 0.6
    acf[2,2,2] = 0.2
    acf[2,1,0] = 0.4

    # generate 3-channel signal from matrix above
    y = cp.mvar_gen(acf,int(10e4))

    # assign static class cp.Mvar to variable mv
    mv = cp.Mvar

    # find best model order
    best, crit = mv._order_akaike(y,15,'vm')
    plt.plot(1+np.arange(len(crit)),crit,'g')
    plt.show()
    print best
    # here we know that this is 3 but in real-life cases
    # we are always uncertain about it

    # now let's fit parameters to the signal
    av, vf = mv.fit(y, best, 'vm')

    # and check whether values are correct +/- 0.01
    print np.allclose(acf, av, 0.01, 0.01)

    # now we can calculate Directed Transfer Function from the data
    dtf = cp.conn.DTF()
    dtfval = dtf.calculate(av, vf, 128)
    # all possible methods are visible in that dictionary:
    print cp.conn.conn_estim_dc.keys()

    cp.plot_conn(dtfval,'DTF values', fs)

Instantaneous
########

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import connectivipy as cp

    # this example reproduce simulation from article:
    # Erla S et all (2009) "Multivariate autoregressive model with 
    #                      instantaneous effects to improve brain 
    #                      connectivity estimation"


    # let's make a matrix from original article

    bcf = np.zeros((4,5,5))
    # matrix shape meaning (k,k,p) k - number of channels,
    # p - order of mvar parameters
    bcf[1,0,0] = 1.58
    bcf[2,0,0] = -0.81
    bcf[0,1,0] = 0.9
    bcf[2,1,1] = -0.01
    bcf[3,1,4] = -0.6
    bcf[1,2,1] = 0.3
    bcf[1,2,2] = 0.8
    bcf[2,2,1] = 0.3
    bcf[2,2,2] = -0.25
    bcf[3,2,1] = 0.3
    bcf[0,3,1] = 0.9
    bcf[1,3,1] = -0.6
    bcf[3,3,1] = 0.3
    bcf[1,4,3] = -0.3
    bcf[2,4,0] = 0.9
    bcf[2,4,3] = -0.3
    bcf[3,4,2] = 0.6

    # now we build a corresponding MVAR process without instantenous effect
    L = np.linalg.inv(np.eye(5)-bcf[0])
    acf = np.zeros((3,5,5))
    for i in xrange(3):
        acf[i] = np.dot(L,bcf[i+1])

    # generate 5-channel signals from matrix above
    signal_inst = cp.mvar_gen_inst(bcf,int(10e4))
    signal = cp.mvar_gen(acf,int(10e4))

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

