.. _tutorial:

Tutorial
==================

All examples and tutorial will go here.

Data class example
########
.. code-block:: python
    
    # Example 1

    import numpy as np 
    import connectivipy as cp
    from connectivipy.mvar.fitting import mvar_gen

    # MVAR model coefficients
    A = np.zeros((2, 5, 5))
    A[0, 0, 0] = 0.95 * 2**0.5
    A[1, 0, 0] = -0.9025
    A[0, 1, 0] = -0.5
    A[1, 2, 1] = 0.4
    A[0, 3, 2] = -0.5
    A[0, 3, 3] = 0.25 * 2**0.5
    A[0, 3, 4] = 0.25 * 2**0.5
    A[0, 4, 3] = -0.25 * 2**0.5
    A[0, 4, 4] = 0.25 * 2**0.5

    # multitrial signal generation
    ysig = np.zeros((5,10**3,5))
    ysig[:,:,0] = mvar_gen(A,10**3)
    ysig[:,:,1] = mvar_gen(A,10**3)
    ysig[:,:,2] = mvar_gen(A,10**3)
    ysig[:,:,3] = mvar_gen(A,10**3)
    ysig[:,:,4] = mvar_gen(A,10**3)

    # connectivity analysis
    data = cp.Data(ysig,128, ["Fp1", "Fp2","Cz", "O1","O2"])

    # you may want to plot data (in multitrial case average along trials
    # is shown)
    data.plot_data()

    # fit mvar using specific algorithm
    data.fit_mvar(2,'yw')

    # you can capture fitted parameters and residual matrix
    ar, vr = data.mvar_coefficients 

    # connectivity estimators
    gdtf_values = data.conn('gdtf')
    #gdtf_significance = data.significance(Nrep=200, alpha=0.05)
    data.plot_conn('gDTF')

    # short time version with default parameters
    pdc_shorttime = data.short_time_conn('pdc', nfft=1, no=10)
    data.plot_short_time_conn("PDC")

How to use specific classes
########
.. code-block:: python
    
    # Example 2

    import numpy as np
    import matplotlib.pyplot as plt
    import connectivipy as cp
    from connectivipy.mvar.fitting import mvar_gen

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
    y = mvar_gen(acf,int(10e4))

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
