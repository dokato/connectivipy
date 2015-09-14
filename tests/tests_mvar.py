# -*- coding: utf-8 -*-
#! /usr/bin/env python

from __future__ import absolute_import
import unittest
import numpy as np 
import connectivipy as cp
from connectivipy.mvar.fitting import *
from connectivipy.mvar.comp import ldl
from connectivipy.mvarmodel import Mvar
from connectivipy.conn import *

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

class MvarTest(unittest.TestCase):

    def test_fitting(self):
        "test MVAR fitting"
        ys = mvar_gen(A, 10**4)
        avm,vvm = vieiramorf(ys, 2)
        ans,vns = nutallstrand(ys, 2)
        ayw,vyw = yulewalker(ys, 2)
        #check dimesions
        self.assertEquals(A.shape, avm.shape)
        self.assertEquals(A.shape, ans.shape)
        self.assertEquals(A.shape, ayw.shape)
        #check values
        self.assertTrue(np.allclose(A, avm, rtol=1e-01, atol=1e-01))
        self.assertTrue(np.allclose(A, ayw, rtol=1e-01, atol=1e-01))
        self.assertTrue(np.allclose(A, ans, rtol=1e-01, atol=0.5))
    
    def test_order_akaike(self):
        m = Mvar()
        ys = mvar_gen(A, 10**4)
        crmin,critpl = m.order_akaike(ys, p_max=20, method='yw')
        self.assertEqual(crmin, 2)

    def test_order_schwartz(self):
        m = Mvar()
        ys = mvar_gen(A, 10**4)
        crmin,critpl = m.order_schwartz(ys, p_max=20, method='yw')
        self.assertEqual(crmin, 2)

    def test_order_hq(self):
        m = Mvar()
        ys = mvar_gen(A, 10**4)
        crmin,critpl = m.order_hq(ys, p_max=20, method='yw')
        self.assertEqual(crmin, 2)

class CompTest(unittest.TestCase):
    def test_ldl(self):
        mactst = np.array([[4,12,-16], [12,37,-43], [-16,-43,98]], dtype=float)
        l, d, lt = ldl(mactst)
        l_sol = np.array([[1, 0, 0],[3, 1, 0],[-4, 5, 1]], dtype=float)
        d_sol = np.array([[4, 0, 0],[0, 1, 0],[0, 0, 9]], dtype=float)
        lt_sol = np.array([[1, 3, -4],[0, 1, 5],[0, 0, 1]], dtype=float)
        np.testing.assert_array_equal(l, l_sol)
        np.testing.assert_array_equal(d, d_sol)
        np.testing.assert_array_equal(lt, lt_sol)

if __name__ == '__main__':
    unittest.main()
