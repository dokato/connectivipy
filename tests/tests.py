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


# from Erla, S. et all (2009)
Ains = np.zeros((4, 5, 5))
Ains[1, 0, 0] = 1.58
Ains[2, 0, 0] = -0.81
Ains[0, 1, 0] = 0.9
Ains[2, 1, 1] = -0.01
Ains[3, 1, 4] = -0.6
Ains[1, 2, 1] = 0.3
Ains[1, 2, 2] = 0.8
Ains[2, 2, 1] = 0.3
Ains[2, 2, 2] = -0.25
Ains[3, 2, 1] = 0.3
Ains[0, 3, 1] = 0.9
Ains[1, 3, 1] = -0.6
Ains[3, 3, 1] = 0.3
Ains[1, 4, 3] = -0.3
Ains[2, 4, 0] = 0.9
Ains[2, 4, 3] = -0.3
Ains[3, 4, 2] = 0.6

class DataTest(unittest.TestCase):
    "test Data class"
    #def test_loading(self):
        #dt = cp.Data('test_data/testsml.raw',data_info='sml')
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
        ys = mvar_gen(A,10**3)
        dat = cp.Data(ys)
        with self.assertRaises(AttributeError):
            dat.conn('dtf')

    def test_conn2(self):
        ys = mvar_gen(A,10**3)
        dat = cp.Data(ys)
        dat.fit_mvar(2,'vm')
        estm = dat.short_time_conn('dtf', nfft=100, no=10)
        stst = dat.short_time_significance(Nrep=100,alpha=0.5, verbose=False)
        self.assertTrue(np.all(stst<=1))

class MvarTest(unittest.TestCase):
    def test_fitting(self):
        "test MVAR fitting"
        ys = mvar_gen(A,10**4)
        avm,vvm = vieiramorf(ys,2)
        ans,vns = nutallstrand(ys,2)
        ayw,vyw = yulewalker(ys,2)
        #check dimesions
        self.assertEquals(A.shape,avm.shape)
        self.assertEquals(A.shape,ans.shape)
        self.assertEquals(A.shape,ayw.shape)
        #check values
        self.assertTrue(np.allclose(A,avm,rtol=1e-01, atol=1e-01))
        self.assertTrue(np.allclose(A,ayw,rtol=1e-01, atol=1e-01))
        self.assertTrue(np.allclose(A,ans,rtol=1e-01, atol=0.5))
    
    def test_orders(self):
        m = Mvar()
        ys = mvar_gen(A,10**4)
        crmin,critpl = m.order_schwartz(ys, p_max=20, method='yw')
        self.assertEqual(crmin, 2)
        crmin,critpl = m.order_akaike(ys, p_max=20, method='yw')
        self.assertEqual(crmin, 2)
        crmin,critpl = m.order_hq(ys, p_max=20, method='yw')
        self.assertEqual(crmin, 2)

class ConnTest(unittest.TestCase):
    "test connectivity class Conn"
    
    def test_spectrum(self):
        ys = mvar_gen(A,10**4)
        avm,vvm = vieiramorf(ys,2)
        a,h,s = spectrum(avm,vvm,512)
        s13 = np.abs(s[:,1,3])
        fq = np.linspace(0,512./2,s.shape[0])
        self.assertAlmostEqual(fq[np.argmax(s13)],65, delta=1.0)

    def test_dtf(self):
        ys = mvar_gen(A,10**4)
        ans,vns = nutallstrand(ys,2)
        dt = DTF() 
        dtf = dt.calculate(ans,vns, 128)
        self.assertTrue(np.allclose(np.sum(np.abs(dtf)**2,axis=2),1))

    def test_pdc(self):
        ys = mvar_gen(A,10**4)
        ans,vns = vieiramorf(ys,2)
        dt = PDC() 
        pdc = dt.calculate(ans,vns, 128)
        self.assertTrue(np.allclose(np.sum(np.abs(pdc)**2,axis=1),1))

    def test_gpdc(self):
        ys = mvar_gen(A,10**4)
        ans,vns = vieiramorf(ys,2)
        dt = gPDC() 
        pdc = dt.calculate(ans,vns, 128)
        self.assertTrue(np.allclose(np.sum(np.abs(pdc)**2,axis=1),1))

    def test_ipdc(self):
        ys = mvar_gen_inst(Ains,10**4)
        ans, vns = yulewalker(ys,4)
        dt = iPDC() 
        ipdc = dt.calculate(ans,vns, 128)
        self.assertTrue(np.allclose(np.sum(np.abs(ipdc)**2,axis=1),1))

    def test_twosided(self):
        gci = GCI()
        psi = PSI()
        gdtf = gDTF()
        ipdc = iPDC()
        self.assertTrue(psi.two_sided)
        self.assertTrue(gci.two_sided)
        self.assertFalse(gdtf.two_sided)
        self.assertFalse(ipdc.two_sided)

if __name__ == '__main__':
    unittest.main()
