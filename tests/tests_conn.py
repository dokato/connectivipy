# -*- coding: utf-8 -*-
#! /usr/bin/env python

from __future__ import absolute_import
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

ys = mvar_gen(A,10**4)

class ConnTest(unittest.TestCase):
    "test connectivity class Conn"
    
    def test_spectrum(self):
        avm,vvm = vieiramorf(ys,2)
        a,h,s = spectrum(avm,vvm,512)
        s13 = np.abs(s[:,1,3])
        fq = np.linspace(0,512./2,s.shape[0])
        self.assertAlmostEqual(fq[np.argmax(s13)],65, delta=1.0)

    def test_dtf(self):
        ans,vns = nutallstrand(ys,2)
        dt = DTF() 
        dtf = dt.calculate(ans,vns, 128)
        self.assertTrue(np.allclose(np.sum(np.abs(dtf)**2,axis=2),1))

    def test_idtf(self):
        ys = mvar_gen_inst(Ains,10**4)
        ans, vns = yulewalker(ys,4)
        dt = iDTF() 
        idtf = dt.calculate(ans,vns, 128)
        self.assertTrue(np.allclose(np.sum(np.abs(idtf)**2,axis=2),1))

    def test_pdc(self):
        ans,vns = vieiramorf(ys,2)
        dt = PDC() 
        pdc = dt.calculate(ans,vns, 128)
        self.assertTrue(np.allclose(np.sum(np.abs(pdc)**2,axis=1),1))

    def test_gpdc(self):
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

    def test_coherency(self):
        fs = 128.
        ch = Coherency()
        chv = ch.calculate(ys, cnfft=100 , cno=10)
        fr = np.linspace(0, int(fs/2), chv.shape[0])
        self.assertTrue(10<fr[np.argmax(chv[:,0,1])]<20)

    def test_psi(self):
        fs = 128.
        ps = PSI()
        psval = ps.calculate(ys, psinfft=200, psino=10)
        self.assertTrue(np.sum(psval[:, 0, 1])==-np.sum(psval[:, 1, 0]))
        self.assertTrue(np.sum(psval[:, 0, 2])==-np.sum(psval[:, 2, 0]))

    def test_gci(self):
        gc = GCI()
        gcval = gc.calculate(ys)
        gc_d = abs(gcval[:,1,0][0])+abs(gcval[:,2,0][0])+abs(gcval[:,2,1][0])
        gc_u = abs(gcval[:,0,1][0])+abs(gcval[:,0,2][0])+abs(gcval[:,1,2][0])
        self.assertTrue(gc_d > gc_u)

    def test_aec(self):
        fs = 256.
        aec = AEC()
        aecval = aec.calculate(ys, fs)
        self.assertTrue(all([all(np.diag(aecval[i])) for i in range(aecval.shape[0])]))
        self.assertEqual(aecval.shape, (5, 5, 5))

    def test_twosided(self):
        gci = GCI()
        psi = PSI()
        gdtf = gDTF()
        ipdc = iPDC()
        self.assertTrue(psi.two_sided)
        self.assertFalse(gci.two_sided)
        self.assertFalse(gdtf.two_sided)
        self.assertFalse(ipdc.two_sided)

if __name__ == '__main__':
    unittest.main()
