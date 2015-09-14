# -*- coding: utf-8 -*-
#! /usr/bin/env python

import unittest
import numpy as np 
import connectivipy as cp
from  connectivipy.mvar.fitting import *
from  connectivipy.mvarmodel import Mvar
from connectivipy.conn import *
import pylab as py

#Parameters from Sameshima, Baccala (2001) Fig. 3a
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

ys = mvar_gen(A,10**3)

class DataTest(unittest.TestCase):
    "test Data class"
    #def test_loading(self):
        #dt = cp.Data('./test_data/testsml.raw',data_info='sml')
        #self.assertEquals(dt.fs,256)
        #self.assertEquals(len(dt.channelnames),2)
        #dd = cp.Data('test_data/m.mat' ,data_info='m')
        #self.assertEquals(dd.data.shape[0],3)
        #self.assertEquals(dd.data.shape[1],5)

    def test_resample(self):
        do = cp.Data(np.random.randn(3,100, 4), fs=10)
        do.resample(5)
        self.assertEquals(do.fs,5)

    def test_conn(self):
        dat = cp.Data(ys)
        with self.assertRaises(AttributeError):
            dat.conn('dtf')

    def test_conn2(self):
        dat = cp.Data(ys)
        dat.fit_mvar(2,'vm')
        estm = dat.short_time_conn('dtf', nfft=100, no=10)
        stst = dat.short_time_significance(Nrep=100, alpha=0.5, verbose=False)
        self.assertTrue(np.all(stst <= 1))

    def test_mvar_calc(self):
        data = cp.Data(ys,128, ["Fp1", "Fp2","Cz", "O1","O2"])
        data.fit_mvar(2,'vm')
        acoef, vcoef = data.mvar_coefficients
        self.assertEquals(acoef.shape, (2, 5, 5))
        self.assertEquals(vcoef.shape, (5, 5))

if __name__ == '__main__':
    unittest.main()
