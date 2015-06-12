# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from abc import ABCMeta, abstractmethod

def spectrum(acoef, vcoef, fs, resolution = None):
    p, k, k = acoef.shape 
    if resolution == None:
        freqs=np.linspace(0,fs/2,512)
    A_z=np.zeros((len(freqs),k,k))+0j
    H_z=np.zeros((len(freqs),k,k))+0j
    S_z=np.zeros((len(freqs),k,k))+0j
    A_z[1:p + 1] = acoef
    A_z = np.eye(k) - np.fft.fft(A_z, axis=0)
    for i in range(len(freqs)):
        H_z[i] = np.linalg.inv(A_z[i])
        S_z[i] = np.dot(np.dot(H_z[i],vcoef), H_z[i].T.conj())
    return A_z, H_z, S_z

class Connect(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate(self):
        pass

    @abstractmethod
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
