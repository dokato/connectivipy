import unittest
import numpy as np 
import connectivipy as cp
from  connectivipy.mvar.fitting import mvar_gen, vieiramorf
from  connectivipy.mvar.fitting import nutallstrand, yulewalker

class DataTest(unittest.TestCase):
    "test data class"
    def test_load(self):
        pass 

class MvarTest(unittest.TestCase):
 
    def test_fitting(self):
        "test mvar fitting"
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
        ys = mvar_gen(A,10**4)
        avm = vieiramorf(ys,2)
        ans = nutallstrand(ys,2)
        ayw = yulewalker(ys,2)
        #check dimesions
        self.assertEquals(A.shape,avm.shape)
        self.assertEquals(A.shape,ans.shape)
        self.assertEquals(A.shape,ayw.shape)
        #check values
        self.assertTrue(np.allclose(A,avm,rtol=1e-01, atol=1e-01))
        self.assertTrue(np.allclose(A,ayw,rtol=1e-01, atol=1e-01))
        self.assertTrue(np.allclose(A,ans,rtol=1e-01, atol=0.5))
 
if __name__ == '__main__':
    unittest.main()
