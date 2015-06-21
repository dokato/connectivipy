# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from abc import ABCMeta, abstractmethod
from mvar.comp import ldl
import pdb

########################################################################
# Spectrum functions:
########################################################################

def spectrumft(acoef, vcoef, fs, resolution = None):
    "not ready to use"
    p, k, k = acoef.shape 
    if resolution == None:
        freqs=np.linspace(0,fs/2,512)
    A_z=np.zeros((len(freqs),k,k),complex)
    H_z=np.zeros((len(freqs),k,k),complex)
    S_z=np.zeros((len(freqs),k,k),complex)
    A_z[1:p + 1] = acoef
    A_z = np.eye(k) - np.fft.rfft(A_z, axis=0)
    for i in xrange(len(freqs)):
        H_z[i] = np.linalg.inv(A_z[i])
        S_z[i] = np.dot(np.dot(H_z[i],vcoef), H_z[i].T.conj())
    return A_z, H_z, S_z

def spectrum(acoef, vcoef, fs, resolution = None):
    "ready to use"
    p, k, k = acoef.shape 
    if resolution == None:
        freqs=np.linspace(0,fs/2,512)
    A_z=np.zeros((len(freqs),k,k),complex)
    H_z=np.zeros((len(freqs),k,k),complex)
    S_z=np.zeros((len(freqs),k,k),complex)
    
    I = np.eye(k,dtype=complex)
    
    for e,f in enumerate(freqs):
        epot = np.zeros((p,1),complex)
        ce = np.exp(-2.j*np.pi*f*(1./fs))
        epot[0] = ce
        for k in xrange(1,p):
            epot[k] = epot[k-1]*ce
        A_z[e] = I - np.sum([epot[x]*acoef[x] for x in range(p)],axis=0)
        #pdb.set_trace()
        H_z[e] = np.linalg.inv(A_z[e])
        S_z[e] = np.dot(np.dot(H_z[e],vcoef), H_z[e].T.conj())
    return A_z, H_z, S_z


def spectrum_b(acoef, vcoef, fs, resolution = None):
    """
    ready to use (i hope)
    from ldlt decomposition
    """
    p, k, k = acoef.shape 
    if resolution == None:
        freqs=np.linspace(0,fs/2,512)
    A_z=np.zeros((len(freqs),k,k),complex)
    B_z=np.zeros((len(freqs),k,k),complex)

    L,U,Lt = ldl(vcoef)
    Linv = np.linalg.inv(L)
    I = np.eye(k,dtype=complex)
    bcoef = np.array([np.dot(Linv, acoef[x]) for x in range(p)])
    b0 = np.eye(k) - Linv
    for e,f in enumerate(freqs):
        epot = np.zeros((p,1),complex)
        ce = np.exp(-2.j*np.pi*f*(1./fs))
        epot[0] = ce
        for k in xrange(1,p):
            epot[k] = epot[k-1]*ce
        B_z[e] = I - b0 - np.sum([epot[x]*bcoef[x] for x in range(p)],axis=0)
    return B_z

########################################################################
# Connectivity classes:
########################################################################

class Connect(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate(self):
        pass

    def short_time(self):
        pass

    @abstractmethod
    def significance(self):
        pass

class ConnectAR(Connect):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit_ar(self):
        pass

class Coherency(Connect):
    pass 

class PSI(Connect):
    pass 

class DTF(ConnectAR):
    """
    Directed transfer function
    Kaminski, M.; Blinowska, K. J. (1991).
    """
    # not too good
    def fit_ar(self, data, order = None, method = 'yw'):
        pass
    
    def calculate(self, Acoef = None, Vcoef = None, fs = None):
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution = None) 
        res, k, k = A_z.shape
        DTF = np.zeros((res,k,k))
        sigma = np.diag(Vcoef)
        for i in xrange(res):
            mH = np.dot(H_z[i],H_z[i].T.conj()).real
            DTF[i] = (np.abs(H_z[i]).T/np.sqrt(np.diag(mH))).T
        return DTF

    def significance(self):
        pass

class PartialCoh(ConnectAR):
    """
    partial coherency
    """

    def fit_ar(self, data, order = None, method = 'yw'):
        pass
    
    def calculate(self, Acoef = None, Vcoef = None, fs = None):
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution = None) 
        res, k, k = A_z.shape
        PC = np.zeros((res,k,k))
        before = np.ones((k,k))
        before[0::2,:]*=-1
        before[:,0::2]*=-1
        for i in xrange(res):
            D_z = np.linalg.inv(S_z[i])
            dd = np.tile(np.diag(D_z),(k,1))
            mD = (dd*dd.T).real
            PC[i] = -1*before*(np.abs(D_z)/np.sqrt(mD))
        return np.abs(PC)

    def significance(self):
        pass

class PDC(ConnectAR):
    """
    PDC
    """
    # not too good
    def fit_ar(self, data, order = None, method = 'yw'):
        pass
    
    def calculate(self, Acoef = None, Vcoef = None, fs = None):
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution = None) 
        res, k, k = A_z.shape
        PDC = np.zeros((res,k,k))
        sigma = np.diag(Vcoef)
        for i in xrange(res):
            mA = np.dot(A_z[i].T.conj(),A_z[i]).real
            PDC[i] = np.abs(A_z[i])/np.sqrt(np.diag(mA))
        return PDC

    def significance(self):
        pass

class dDTF(ConnectAR):
    """
    dDTF
    Korzeniewska, A.et. all. Determination of information flow direction 
    among brain structures by a modified directed transfer function (dDTF) 
    method. J. Neurosci. Methods 125, 195–207 (2003).
    """

    def fit_ar(self, data, order = None, method = 'yw'):
        pass

    def calculate(self, Acoef = None, Vcoef = None, fs = None):
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution = None) 
        res, k, k = A_z.shape
        mH = np.zeros((res,k,k))
        for i in xrange(res):
            mH[i] = np.abs(np.dot(H_z[i],H_z[i].T.conj()))
        mHsum = np.sum(mH, axis=0)
        print '>'*8, mHsum
        print mH[30]
        dDTF = np.zeros((res,k,k))
        before = np.ones((k,k))
        before[0::2,:]*=-1
        before[:,0::2]*=-1
        for i in xrange(res):
            D_z = np.linalg.inv(S_z[i])
            dd = np.tile(np.diag(D_z),(k,1))
            mD = (dd*dd.T).real
            PC = np.abs(-1*before*(np.abs(D_z)/np.sqrt(mD)))
            dDTF[i] = (np.abs(H_z[i]).T/np.sqrt(np.diag(mHsum))).T
        return dDTF

    def significance(self):
        pass

class iPDC(ConnectAR):
    """
    !!! warning not tested !!!
    iPDC
    Erla, S. et all Multivariate Autoregressive Model with Instantaneous
    Effects to Improve Brain Connectivity Estimation. 
    Int. J. Bioelectromagn. 11, 74–79 (2009).
    """

    def fit_ar(self, data, order = None, method = 'yw'):
        pass
    
    def calculate(self, Acoef = None, Vcoef = None, fs = None):
        B_z = spectrum_b(Acoef, Vcoef, fs, resolution = None) 
        res, k, k = B_z.shape
        PDC = np.zeros((res,k,k))
        sigma = np.diag(Vcoef)
        for i in xrange(res):
            mB = np.dot(B_z[i].T.conj(),B_z[i]).real
            PDC[i] = np.abs(B_z[i])/np.sqrt(np.diag(mB))
        return PDC

    def significance(self):
        pass
