import unittest
import numpy as np 
import connectivipy as cp
from  connectivipy.mvar.fitting import mvar_gen, mvar_gen_inst
from  connectivipy.mvar.fitting import vieiramorf, nutallstrand, yulewalker
from  connectivipy.mvarmodel import Mvar
from connectivipy.conn import ConnectAR, spectrum, spectrumft, DTF, PDC
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
A2 = np.zeros((4, 5, 5))
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
    "test data class"
    def test_load(self):
        pass 

class MvarTest(unittest.TestCase):
    def test_fitting(self):
        "test mvar fitting"
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
        crmin,critpl = m._order_schwartz(ys, p_max = 20, method = 'yw')
        print '****', crmin
        #py.plot(critpl)
        #py.show()

class ConnTest(unittest.TestCase):
    def test_spectrum(self):
        ys = mvar_gen(A,10**4)
        avm,vvm = vieiramorf(ys,2)
        a,h,s = spectrum(avm,vvm,512)
        s13 = np.abs(s[:,1,3])
        fq = np.linspace(0,512./2,s.shape[0])
        self.assertAlmostEqual(fq[np.argmax(s13)],64, places=0)

    def test_dtf(self):
        ys = mvar_gen(A,10**4)
        ans,vns = nutallstrand(ys,2)
        dt = DTF() 
        dtf = dt.calculate(ans,vns, 128)
        self.assertTrue(np.allclose(np.sum(np.abs(dtf)**2,axis=2),1))

    def test_gdtf(self):
        ys = mvar_gen(A,10**4)
        ans,vns = nutallstrand(ys,2)
        dt = gDTF() 
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

if __name__ == '__main__':
    unittest.main()
