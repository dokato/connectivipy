import unittest
import numpy as np 
import connectivipy as cp
from  connectivipy.mvar.fitting import mvar_gen, vieiramorf
from  connectivipy.mvar.fitting import nutallstrand, yulewalker
from  connectivipy.mvarmodel import Mvar
from connectivipy.conn import ConnectAR, spectrum, spectrumft, DTF, PDC
import pylab as py

#Parameters from Sameshima, Baccala (2001) Fig. 3a
A = np.zeros((2, 5, 5))
A[0, 0, 0] = 0.95 * (2)**0.5
A[1, 0, 0] = -0.9025
A[0, 1, 0] = -0.5
A[1, 2, 1] = 0.4
A[0, 3, 2] = -0.5
A[0, 3, 3] = 0.25 * (2)**0.5
A[0, 3, 4] = 0.25 * (2)**0.5
A[0, 4, 3] = -0.25 * (2)**0.5
A[0, 4, 4] = 0.25 * (2)**0.5

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

if __name__ == '__main__':
    unittest.main()
