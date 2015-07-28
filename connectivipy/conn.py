# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from abc import ABCMeta, abstractmethod
from mvar.comp import ldl
from mvarmodel import Mvar
import pdb
import scipy.stats as st

########################################################################
# Spectrum functions:
########################################################################

def spectrumft(acoef, vcoef, fs=1, resolution=None):
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

def spectrum(acoef, vcoef, fs=1, resolution=None):
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
        H_z[e] = np.linalg.inv(A_z[e])
        S_z[e] = np.dot(np.dot(H_z[e],vcoef), H_z[e].T.conj())
    return A_z, H_z, S_z


def spectrum_inst(acoef, vcoef, fs=1, resolution=None):
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
    
    def short_time(self, winlen=64, no=16):
        pass

    def significance(self, data, Nrep=10, alpha=0.05, **params):
        if len(data.shape)>2:
            signific = self.bootstrap(data, Nrep=10, alpha=alpha, **params)
        else:
            signific = self.surrogate(data, Nrep=10, alpha=alpha, **params)
        return signific

    def levels(self, signi, alpha, k):
        ficance = np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                ficance[i][j] = np.max(st.scoreatpercentile(signi[:,:,i,j], alpha*100, axis=1))
        return ficance
        
    def __calc_multitrial(self, data, **params):
        trials = data.shape[2]
        chosen = np.random.randint(trials,size=trials)
        bc = np.bincount(chosen)
        idxbc = np.nonzero(bc)[0]
        flag = True
        for num, occurence in zip(idxbc, bc[idxbc]):
            if occurence>0:
                trdata = data[:,:,num]
                if flag:
                    rescalc = self.calculate(trdata, **params)*occurence                    
                    flag = False
                    continue
                rescalc += self.calculate(trdata, **params)*occurence
        return rescalc/trials

    def bootstrap(self, data, Nrep=10, alpha=0.05, **params):
        for i in xrange(Nrep):
            print '.',
            if i==0:
                tmpsig = self.__calc_multitrial(data, **params)
                fres, k, k = tmpsig.shape
                signi = np.zeros((Nrep, fres, k, k))
                signi[i] = tmpsig
            else:
                signi[i] = self.__calc_multitrial(data, **params)
        print '|'
        return self.levels(signi, alpha, k)

    def surrogate(self, data, Nrep = 10, alpha=0.05, **params):
        k, N = data.shape
        shdata = data.copy()
        for i in xrange(Nrep):
            print '.',
            map(np.random.shuffle, shdata)
            if i==0:
                rtmp = self.calculate(data, **params)
                reskeeper = np.zeros((Nrep, rtmp.shape[0], k, k))
                reskeeper[i] = rtmp
                continue
            reskeeper[i] = self.calculate(data, **params)
        print '|'
        return self.levels(reskeeper, alpha, k)

class ConnectAR(Connect):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit_ar(self):
        pass

    def __calc_multitrial(self, arrs, vrrs, fs, resolution):
        trials = arrs.shape[0]
        chosen = np.random.randint(trials,size=trials)
        bc = np.bincount(chosen)
        idxbc = np.nonzero(bc)[0]
        flag = True
        for num, occurence in zip(idxbc, bc[idxbc]):
            if occurence>0:
                if flag:
                    rescalc = self.calculate(arrs[num], vrrs[num], fs, resolution)*occurence
                    flag = False
                    continue
                rescalc += self.calculate(arrs[num], vrrs[num], fs, resolution)*occurence
        return rescalc/trials

    def bootstrap(self, arrs, vrrs, Nrep = 10, alpha=0.05, fs=1, **params):
        resolution = None
        if params.has_key('resolution') and params['resolution']:
            resolution = params['resolution']
        for i in xrange(Nrep):
            print '.',
            if i==0:
                tmpsig = self.__calc_multitrial(arrs, vrrs, fs, resolution)
                fres, k, k = tmpsig.shape
                signi = np.zeros((Nrep, fres, k, k))
                signi[i] = tmpsig
            else:
                signi[i] = self.__calc_multitrial(arrs, vrrs, fs, resolution)
        print '|'
        return self.levels(signi, alpha, k)

    def surrogate(self, data, method, Nrep = 10, alpha=0.05, order=None, fs=1, **params):
        shdata = data.copy()
        k, N = data.shape
        resolution = None
        if params.has_key('resolution') and params['resolution']:
            resolution = params['resolution']
        for i in xrange(Nrep):
            print '.',
            map(np.random.shuffle, shdata)
            ar, vr = Mvar().fit(shdata, order, method)
            if i==0:
                rtmp = self.calculate(ar, vr, fs, resolution)
                reskeeper = np.zeros((Nrep, rtmp.shape[0], k, k))
                reskeeper[i] = rtmp
                continue
            reskeeper[i] = self.calculate(ar, vr, fs, resolution)
        print '|'
        return self.levels(signi, alpha, k)

############################
# MVAR based methods:

def dtf_fun(Acoef, Vcoef, fs, resolution, generalized=False):
    A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution = resolution) 
    res, k, k = A_z.shape
    DTF = np.zeros((res,k,k))
    if generalized:
        sigma = np.diag(Vcoef)
    else:
        sigma = np.ones(k)
    for i in xrange(res):
        mH = sigma*np.dot(H_z[i],H_z[i].T.conj()).real
        DTF[i] = (np.sqrt(sigma)* np.abs(H_z[i]))/np.sqrt(np.diag(mH)).reshape((k,1))
    return DTF

def pdc_fun(Acoef, Vcoef, fs, resolution, generalized=False):
    A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution = resolution) 
    res, k, k = A_z.shape
    PDC = np.zeros((res,k,k))
    sigma = np.diag(Vcoef)
    for i in xrange(res):
        mA = (1./sigma[:, None])*np.dot(A_z[i].T.conj(),A_z[i]).real
        PDC[i] = np.abs(A_z[i] / np.sqrt(sigma))/np.sqrt(np.diag(mA))
    return PDC

class PartialCoh(ConnectAR):
    """
    partial coherency
    """
    
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=None):
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution=resolution) 
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

class DTF(ConnectAR):
    """
    Directed transfer function
    Kaminski, M.; Blinowska, K. J. (1991).
    """

    def calculate(self, Acoef = None, Vcoef = None, fs = None, resolution = None):
        return dtf_fun(Acoef, Vcoef, fs, resolution)

class PDC(ConnectAR):
    """
    PDC
    """
    
    def calculate(self, Acoef = None, Vcoef = None, fs = None, resolution = None):
        return pdc_fun(Acoef, Vcoef, fs, resolution)

class gPDC(ConnectAR):
    """
    generalized PDC
    """
    # not too good
    def fit_ar(self, data, order = None, method = 'yw'):
        pass
    
    def calculate(self, Acoef = None, Vcoef = None, fs = None, resolution = None):
        return pdc_fun(Acoef, Vcoef, fs, resolution, generalized=True)

class gDTF(ConnectAR):
    """
    Directed transfer function
    Kaminski, M.; Blinowska, K. J. (1991).
    """
    # not too good
    def fit_ar(self, data, order = None, method = 'yw'):
        pass
    
    def calculate(self, Acoef = None, Vcoef = None, fs = None, resolution = None):
        return dtf_fun(Acoef, Vcoef, fs, resolution, generalized=True)

class ffDTF(ConnectAR):
    """
    full frequency DTF
    Korzeniewska, A.et. all. Determination of information flow direction 
    among brain structures by a modified directed transfer function (dDTF) 
    method. J. Neurosci. Methods 125, 195–207 (2003).
    """

    def fit_ar(self, data, order = None, method = 'yw'):
        pass

    def calculate(self, Acoef = None, Vcoef = None, fs = None, resolution = None):
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution = resolution) 
        res, k, k = A_z.shape
        mH = np.zeros((res,k,k))
        for i in xrange(res):
            mH[i] = np.abs(np.dot(H_z[i],H_z[i].T.conj()))
        mHsum = np.sum(mH, axis=0)
        ffDTF = np.zeros((res,k,k))
        for i in xrange(res):
            ffDTF[i] = (np.abs(H_z[i]).T/np.sqrt(np.diag(mHsum))).T
        return ffDTF

class dDTF(ConnectAR):
    """
    dDTF
    Korzeniewska, A.et. all. Determination of information flow direction 
    among brain structures by a modified directed transfer function (dDTF) 
    method. J. Neurosci. Methods 125, 195–207 (2003).
    """

    def fit_ar(self, data, order = None, method = 'yw'):
        pass

    def calculate(self, Acoef = None, Vcoef = None, fs = None, resolution = None):
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution = resolution) 
        res, k, k = A_z.shape
        mH = np.zeros((res,k,k))
        for i in xrange(res):
            mH[i] = np.abs(np.dot(H_z[i],H_z[i].T.conj()))
        mHsum = np.sum(mH, axis=0)
        dDTF = np.zeros((res,k,k))
        before = np.ones((k,k))
        before[0::2,:]*=-1
        before[:,0::2]*=-1
        for i in xrange(res):
            D_z = np.linalg.inv(S_z[i])
            dd = np.tile(np.diag(D_z),(k,1))
            mD = (dd*dd.T).real
            PC = np.abs(-1*before*(np.abs(D_z)/np.sqrt(mD)))
            dDTF[i] = PC*(np.abs(H_z[i]).T/np.sqrt(np.diag(mHsum))).T
        return dDTF

class iPDC(ConnectAR):
    """
    iPDC
    Erla, S. et all Multivariate Autoregressive Model with Instantaneous
    Effects to Improve Brain Connectivity Estimation. 
    Int. J. Bioelectromagn. 11, 74–79 (2009).
    """

    def fit_ar(self, data, order = None, method = 'yw'):
        pass
    
    def calculate(self, Acoef = None, Vcoef = None, fs = None, resolution = None):
        B_z = spectrum_inst(Acoef, Vcoef, fs, resolution = resolution) 
        res, k, k = B_z.shape
        PDC = np.zeros((res,k,k))
        sigma = np.diag(Vcoef)
        for i in xrange(res):
            mB = np.dot(B_z[i].T.conj(),B_z[i]).real
            PDC[i] = np.abs(B_z[i])/np.sqrt(np.diag(mB))
        return PDC

class iDTF(ConnectAR):
    """
        ????
    """
    # not too good
    def fit_ar(self, data, order = None, method = 'yw'):
        pass
    
    def calculate(self, Acoef = None, Vcoef = None, fs = None, resolution = None):
        B_z = spectrum_inst(Acoef, Vcoef, fs, resolution = resolution) 
        res, k, k = B_z.shape
        DTF = np.zeros((res,k,k))
        for i in xrange(res):
            Hb_z[e] = np.linalg.inv(B_z[e])
            mH = np.dot(Hb_z[i],Hb_z[i].T.conj()).real
            DTF[i] = np.abs(Hb_z[i])/np.sqrt(np.diag(mH)).reshape((k,1))
        return DTF

############################
# Fourier Transform based methods:

class Coherency(Connect):
    def calculate(self, data, nfft=None, no=0, window=np.hanning, im=False):
        k, N = data.shape 
        if not nfft:
            nfft = int(N/4)
        if not no:
            no = int(N/10)
        winarr = window(nfft)
        slices = xrange(0, N, int(nfft-no))
        ftsliced = np.zeros((len(slices), k, int(nfft/2)+1), complex)
        for e,i in enumerate(slices):
            if i+nfft>=N:
                datzer = np.concatenate((data[:,i:i+nfft],np.zeros((k,i+nfft-N))),axis=1)
                ftsliced[e] = np.fft.rfft(datzer*winarr, axis=1)
            else:
                ftsliced[e] = np.fft.rfft(data[:,i:i+nfft]*winarr, axis=1)
        ctop = np.zeros((len(slices), k, k, int(nfft/2)+1), complex)
        cdown = np.zeros((len(slices), k, int(nfft/2)+1))
        for i in xrange(len(slices)):
            c1 = ftsliced[i,:,:].reshape((k, 1, int(nfft/2)+1))
            c2 = ftsliced[i,:,:].conj().reshape((1, k, int(nfft/2)+1))
            ctop[i] = c1*c2
            cdown[i] = np.abs(ftsliced[i,:,:])**2
        cd1  = np.mean(cdown,axis=0).reshape((k, 1, int(nfft/2)+1))
        cd2  = np.mean(cdown,axis=0).reshape((1, k, int(nfft/2)+1))
        cdwn = cd1*cd2
        coh  = np.mean(ctop,axis=0)/np.sqrt(cdwn)
        if not im:
            coh = np.abs(coh)
        return coh.T

class PSI(Connect):
    def calculate(self, data, band_width = 4, nfft=None, no=0, window=np.hanning):
        k, N = data.shape 
        coh = Coherency()
        cohval = coh.calculate(data, nfft=nfft, no=no, window=window, im=True)
        fq_bands = np.arange(0, int(nfft/2)+1, band_width)
        psi = np.zeros((len(fq_bands)-1,k,k))
        for f in xrange(len(fq_bands)-1):
            ctmp = cohval[fq_bands[f]:fq_bands[f+1],:,:]
            psi[f] = np.imag(np.sum(ctmp[:-1,:,:].conj()*ctmp[1:,:,:], axis=0))
        #full_psi = np.imag(np.sum(cohval[:-1,:,:].conj()*cohval[1:,:,:]))
        return psi

class GC(Connect):
    def calculate(self, data, order, method):
        k, N = data.shape
        arfull, vrfull = Mvar().fit(data, order, method)
        gcval = np.zeros((k, k))
        for i in xrange(k):
            arix = [j for j in xrange(k) if i!=j]
            ar_i, vr_i = Mvar().fit(data[arix,:], order, method)
            tmpv = np.abs(vrfull)**2/np.abs(vr_i)**2
            gcval[i, :] = np.log(tmpv[i, :])
        return gcval

conn_estim_dc = { 'dtf'  : DTF,
                  'pdc'  : PDC,
                  'ipdc' : iPDC,
                  'psi'  : PSI,
                  'ffdtf': ffDTF,
                  'ddtf' : dDTF,
                  'gdtf' : gDTF,
                  'gpdc' : gPDC,
                  'pcoh' : PartialCoh,
                  'coh'  : Coherency,
                  'gc'   : GC,
                }
