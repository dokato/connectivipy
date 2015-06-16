# -*- coding: utf-8 -*-
#! /usr/bin/env python

from data import Data
from conn import ConnectAR, spectrum, spectrumft
from mvarmodel import Mvar

# delete it later:
import numpy as np 
import pylab as py
from mvar.fitting import mvar_gen, vieiramorf, nutallstrand

__version__ = '0.05'


#spectrum tests:
if __name__ == '__main__':
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


    ys = mvar_gen(A,10**4)
    avm,vvm = vieiramorf(ys,2)

    a,h,s = spectrum(avm,vvm,512)
    print np.max(np.abs(s[:,1,3]))
    s13 = np.abs(s[:,1,3])
    fq = np.linspace(0,512./2,s.shape[0])
    print fq[np.argmax(s13)]
    py.plot(fq, np.abs(s[:,1,3]) )
    py.show()
