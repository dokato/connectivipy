# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from abc import ABCMeta, abstractmethod
import pdb

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
        import pdb
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
