# -*- coding: utf-8 -*-
#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as st
from abc import ABCMeta, abstractmethod
from .mvar.comp import ldl
from .mvarmodel import Mvar

import six
from six.moves import map
from six.moves import range
from six.moves import zip

########################################################################
# Spectrum functions:
########################################################################


def spectrum(acoef, vcoef, fs=1, resolution=100):
    """
    Generating data point from matrix *A* with MVAR coefficients.
    Args:
      *acoef* : numpy.array
          array of shape (k, k, p) where *k* is number of channels and
          *p* is a model order.
      *vcoef* : numpy.array
          prediction error matrix (k, k)
      *fs* = 1 : int
          sampling rate
      *resolution* = 100 : int
          number of spectrum data points
    Returns:
      *A_z* : numpy.array
          z-transformed A(f) complex matrix in shape (*resolution*, k, k)
      *H_z* : numpy.array
          inversion of *A_z*
      *S_z* : numpy.array
          spectrum matrix (*resolution*, k, k)
    References:
    .. [1] K. J. Blinowska, R. Kus, M. Kaminski (2004) “Granger causality
           and information flow in multivariate processes”
           Physical Review E 70, 050902.
    """
    p, k, k = acoef.shape
    freqs = np.linspace(0, fs*0.5, resolution)
    A_z = np.zeros((len(freqs), k, k), complex)
    H_z = np.zeros((len(freqs), k, k), complex)
    S_z = np.zeros((len(freqs), k, k), complex)

    I = np.eye(k, dtype=complex)
    for e, f in enumerate(freqs):
        epot = np.zeros((p, 1), complex)
        ce = np.exp(-2.j*np.pi*f*(1./fs))
        epot[0] = ce
        for k in range(1, p):
            epot[k] = epot[k-1]*ce
        A_z[e] = I - np.sum([epot[x]*acoef[x] for x in range(p)], axis=0)
        H_z[e] = np.linalg.inv(A_z[e])
        S_z[e] = np.dot(np.dot(H_z[e], vcoef), H_z[e].T.conj())
    return A_z, H_z, S_z


def spectrum_inst(acoef, vcoef, fs=1, resolution=100):
    """
    Generating data point from matrix *A* with MVAR coefficients taking
    into account zero-lag effects.
    Args:
      *acoef* : numpy.array
          array of shape (k, k, p+1) where *k* is number of channels and
          *p* is a model order. acoef[0] - is (k, k) matrix for zero lag,
          acoef[1] for one data point lag and so on.
      *vcoef* : numpy.array
          prediction error matrix (k, k)
      *fs* = 1 : int
          sampling rate
      *resolution* = 100 : int
          number of spectrum data points
    Returns:
      *A_z* : numpy.array
          z-transformed A(f) complex matrix in shape (*resolution*, k, k)
      *H_z* : numpy.array
          inversion of *A_z*
      *S_z* : numpy.array
          spectrum matrix (*resolution*, k, k)
    References:
    .. [1] Erla S. et all, Multivariate Autoregressive Model with
           Instantaneous Effects to Improve Brain Connectivity Estimation,
           Int. J. Bioelectromagn. 11, 74–79 (2009).
    """
    p, k, k = acoef.shape
    freqs = np.linspace(0, fs/2, resolution)
    B_z = np.zeros((len(freqs), k, k), complex)
    L, U, Lt = ldl(vcoef)
    Linv = np.linalg.inv(L)
    I = np.eye(k, dtype=complex)
    bcoef = np.array([np.dot(Linv, acoef[x]) for x in range(p)])
    b0 = np.eye(k) - Linv
    for e, f in enumerate(freqs):
        epot = np.zeros((p, 1), complex)
        ce = np.exp(-2.j*np.pi*f*(1./fs))
        epot[0] = ce
        for k in range(1, p):
            epot[k] = epot[k-1]*ce
        B_z[e] = I - b0 - np.sum([epot[x]*bcoef[x] for x in range(p)], axis=0)
    return B_z

########################################################################
# Connectivity classes:
########################################################################


class Connect(six.with_metaclass(ABCMeta, object)):
    """
    Abstract class governing calculation of various connectivity estimators
    with concrete methods: *short_time*, *significance* and
    abstract *calculate*.
    """

    def __init__(self):
        self.values_range = [None, None]  # normalization bands
        self.two_sided = False  # only positive, or also negative values

    @abstractmethod
    def calculate(self):
        """Abstract method to calculate values of estimators from specific
        parameters"""
        pass

    def short_time(self, data, nfft=None, no=None, **params):
        """
        Short-tme version of estimator, where data is windowed into parts
        of length *nfft* and overlap *no*. *params* catch additional
        parameters specific for estimator.
        Args:
          *data* : numpy.array
              data matrix
          *nfft* = None : int
              window length (if None it's N/5)
          *no* = None : int
              overlap length (if None it's N/10)
          *params* :
              additional parameters specific for chosen estimator
        Returns:
          *stvalues* : numpy.array
              short time values (time points, frequency, k, k), where k
              is number of channels
        """
        assert nfft > no, "overlap must be smaller than window"
        if data.ndim > 2:
            k, N, trls = data.shape
        else:
            k, N = data.shape
            trls = 0
        if not nfft:
            nfft = int(N/5)
        if not no:
            no = int(N/10)
        slices = range(0, N, int(nfft-no))
        for e, i in enumerate(slices):
            if i+nfft >= N:
                if trls:
                    datcut = np.concatenate((data[:, i:i+nfft], np.zeros((k, i+nfft-N, trls))), axis=1)
                else:
                    datcut = np.concatenate((data[:, i:i+nfft], np.zeros((k, i+nfft-N))), axis=1)
            else:
                datcut = data[:, i:i+nfft]
            if e == 0:
                rescalc = self.calculate(datcut, **params)
                stvalues = np.zeros((len(slices), rescalc.shape[0], k, k))
                stvalues[e] = rescalc
                continue
            stvalues[e] = self.calculate(datcut, **params)
        return stvalues

    def short_time_significance(self, data, Nrep=10, alpha=0.05,
                                nfft=None, no=None, verbose=True, **params):
        """
        Significance of short-tme versions of estimators. It base on
        bootstrap :func:`Connect.bootstrap` for multitrial case and
        surrogate data :func:`Connect.surrogate` for one trial.
        Args:
          *data* : numpy.array
              data matrix
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
          *nfft* = None : int
              window length (if None it's N/5)
          *no* = None : int
              overlap length (if None it's N/10)
          *verbose* = True : bool
            if True it prints dot on every realization, if False it's
            quiet.
          *params* :
              additional parameters specific for chosen estimator
        Returns:
          *signi_st* : numpy.array
              short time significance values in shape of
              - (tp, k, k) for one sided estimator
              - (tp, 2, k, k) for two sided
              where k is number of channels and tp number of time points
        """
        assert nfft > no, "overlap must be smaller than window"
        if data.ndim > 2:
            k, N, trls = data.shape
        else:
            k, N = data.shape
            trls = 0
        if not nfft:
            nfft = int(N/5)
        if not no:
            no = int(N/10)
        slices = range(0, N, int(nfft-no))
        if self.two_sided:
            signi_st = np.zeros((len(slices), 2, k, k))
        else:
            signi_st = np.zeros((len(slices), k, k))
        for e, i in enumerate(slices):
            if i+nfft >= N:
                if trls:
                    datcut = np.concatenate((data[:, i:i+nfft], np.zeros((k, i+nfft-N, trls))), axis=1)
                else:
                    datcut = np.concatenate((data[:, i:i+nfft], np.zeros((k, i+nfft-N))), axis=1)
            else:
                datcut = data[:, i:i+nfft]
            signi_st[e] = self.significance(datcut, Nrep=Nrep,
                                            alpha=alpha, verbose=verbose, **params)
        return signi_st

    def significance(self, data, Nrep=10, alpha=0.05, verbose=True, **params):
        """
        Significance of connectivity estimators. It base on
        bootstrap :func:`Connect.bootstrap` for multitrial case and
        surrogate data :func:`Connect.surrogate` for one trial.
        Args:
          *data* : numpy.array
              data matrix
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
          *verbose* = True : bool
            if True it prints dot on every realization, if False it's
            quiet.
          *params* :
              additional parameters specific for chosen estimator
        Returns:
          *signific* : numpy.array
              significance values, check :func:`Connect.levels`
        """
        if data.ndim > 2:
            signific = self.bootstrap(data, Nrep=10, alpha=alpha, verbose=verbose, **params)
        else:
            signific = self.surrogate(data, Nrep=10, alpha=alpha, verbose=verbose, **params)
        return signific

    def levels(self, signi, alpha, k):
        """
        Levels of significance
        Args:
          *signi* : numpy.array
              bootstraped values of each channel
          *alpha* : float
              type I error rate (significance level) - from 0 to 1
              - (1-*alpha*) for onesided estimators (e.g. class:`DTF`)
              - *alpha* and (1-*alpha*) for twosided (e.g. class:`PSI`)
          *k* : int
              number of channels
        Returns:
          *ficance* : numpy.array
              maximal value throughout frequency of score at percentile
              at level 1-*alpha*
              - (k, k) for one sided estimator
              - (2, k, k) for two sided
        """
        if self.two_sided:
            ficance = np.zeros((2, k, k))
        else:
            ficance = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if self.two_sided:
                    ficance[0][i][j] = np.min(st.scoreatpercentile(signi[:, :, i, j], alpha*100, axis=0))
                    ficance[1][i][j] = np.max(st.scoreatpercentile(signi[:, :, i, j], (1-alpha)*100, axis=0))
                else:
                    #import pdb; pdb.set_trace()
                    ficance[i][j] = np.min(st.scoreatpercentile(signi[:, :, i, j], (1-alpha)*100, axis=0))
        return ficance

    def __calc_multitrial(self, data, **params):
        "Calc multitrial averaged estimator for :func:`Connect.bootstrap`"
        trials = data.shape[2]
        chosen = np.random.randint(trials, size=trials)
        bc = np.bincount(chosen)
        idxbc = np.nonzero(bc)[0]
        flag = True
        for num, occurence in zip(idxbc, bc[idxbc]):
            if occurence > 0:
                trdata = data[:, :, num]
                if flag:
                    rescalc = self.calculate(trdata, **params)*occurence
                    flag = False
                    continue
                rescalc += self.calculate(trdata, **params)*occurence
        return rescalc/trials

    def bootstrap(self, data, Nrep=100, alpha=0.05, verbose=True, **params):
        """
        Bootstrap - random sampling with replacement of trials.
        Args:
          *data* : numpy.array
              multichannel data matrix
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
          *verbose* = True : bool
            if True it prints dot on every realization, if False it's
            quiet.
          *params* :
              additional parameters specific for chosen estimator
        Returns:
          *levelsigni* : numpy.array
              significance values, check :func:`Connect.levels`
        """
        for i in range(Nrep):
            if verbose:
                print('.', end=' ')
            if i == 0:
                tmpsig = self.__calc_multitrial(data, **params)
                fres, k, k = tmpsig.shape
                signi = np.zeros((Nrep, fres, k, k))
                signi[i] = tmpsig
            else:
                signi[i] = self.__calc_multitrial(data, **params)
        if verbose:
            print('|')
        return self.levels(signi, alpha, k)

    def surrogate(self, data, Nrep=100, alpha=0.05, verbose=True, **params):
        """
        Surrogate data testing. Mixing data points in each channel.
        Significance level in calculated over all *Nrep* surrogate sets.
        Args:
          *data* : numpy.array
              multichannel data matrix
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
          *verbose* = True : bool
            if True it prints dot on every realization, if False it's
            quiet.
          *params* :
              additional parameters specific for chosen estimator
        Returns:
          *levelsigni* : numpy.array
              significance values, check :func:`Connect.levels`
        """
        k, N = data.shape
        shdata = data.copy()
        for i in range(Nrep):
            if verbose:
                print('.', end=' ')
            for ch in range(k):
                np.random.shuffle(shdata[ch,:])
            if i == 0:
                rtmp = self.calculate(shdata, **params)
                reskeeper = np.zeros((Nrep, rtmp.shape[0], k, k))
                reskeeper[i] = rtmp
                continue
            reskeeper[i] = self.calculate(shdata, **params)
        if verbose:
            print('|')
        return self.levels(reskeeper, alpha, k)


class ConnectAR(six.with_metaclass(ABCMeta, Connect)):
    """
    Inherits from *Connect* class and governs calculation of various
    connectivity estimators basing on MVAR model methods. It overloads
    *short_time*, *significance* methods but *calculate* remains abstract.
    """

    def __init__(self):
        super(ConnectAR, self).__init__()
        self.values_range = [0, 1]

    def short_time(self, data, nfft=None, no=None, mvarmethod='yw',
                   order=None, resol=None, fs=1):
        """
        It overloads :class:`ConnectAR` method :func:`Connect.short_time`.
        Short-tme version of estimator, where data is windowed into parts
        of length *nfft* and overlap *no*. *params* catch additional
        parameters specific for estimator.
        Args:
          *data* : numpy.array
              data matrix
          *nfft* = None : int
              window length (if None it's N/5)
          *no* = None : int
              overlap length (if None it's N/10)
          *mvarmethod* = 'yw' :
              MVAR parameters estimation method
            all avaiable methods you can find in *fitting_algorithms*
          *order* = None:
              MVAR model order; it None, it is set automatically basing
              on default criterion.
          *resol* = None:
              frequency resolution; if None, it is 100.
          *fs* = 1 :
              sampling frequency
        Returns:
          *stvalues* : numpy.array
              short time values (time points, frequency, k, k), where k
              is number of channels
        """
        assert nfft > no, "overlap must be smaller than window"
        if data.ndim > 2:
            k, N, trls = data.shape
        else:
            k, N = data.shape
            trls = 0
        if not nfft:
            nfft = int(N/5)
        if not no:
            no = int(N/10)
        slices = range(0, N, int(nfft-no))
        for e, i in enumerate(slices):
            if i+nfft >= N:
                if trls:
                    datcut = np.concatenate((data[:, i:i+nfft], np.zeros((k, i+nfft-N, trls))), axis=1)
                else:
                    datcut = np.concatenate((data[:, i:i+nfft], np.zeros((k, i+nfft-N))), axis=1)
            else:
                datcut = data[:, i:i+nfft]
            ar, vr = Mvar().fit(datcut, order, mvarmethod)
            if e == 0:
                rescalc = self.calculate(ar, vr, fs, resol)
                stvalues = np.zeros((len(slices), rescalc.shape[0], k, k))
                stvalues[e] = rescalc
                continue
            stvalues[e] = self.calculate(ar, vr, fs, resol)
        return stvalues

    def short_time_significance(self, data, Nrep=100, alpha=0.05, method='yw',
                                order=None, fs=1, resolution=None,
                                nfft=None, no=None, verbose=True, **params):
        """
        Significance of short-tme versions of estimators. It base on
        bootstrap :func:`ConnectAR.bootstrap` for multitrial case and
        surrogate data :func:`ConnectAR.surrogate` for one trial.
        Args:
          *data* : numpy.array
              data matrix
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
          *method* = 'yw': str
            method of MVAR parameters estimation
            all avaiable methods you can find in *fitting_algorithms*
          *order* = None : int
            MVAR model order, if None, it's chosen using default criterion
          *fs* = 1 : int
              sampling frequency
          *resolution* = None : int
              resolution (if None, it's 100 points)
          *nfft* = None : int
              window length (if None it's N/5)
          *no* = None : int
              overlap length (if None it's N/10)
          *verbose* = True : bool
            if True it prints dot on every realization, if False it's
            quiet.
          *params* :
              additional parameters specific for chosen estimator
        Returns:
          *signi_st* : numpy.array
              short time significance values in shape of
              - (tp, k, k) for one sided estimator
              - (tp, 2, k, k) for two sided
              where k is number of channels and tp number of time points
        """
        assert nfft > no, "overlap must be smaller than window"
        if data.ndim > 2:
            k, N, trls = data.shape
        else:
            k, N = data.shape
            trls = 0
        if not nfft:
            nfft = int(N/5)
        if not no:
            no = int(N/10)
        slices = range(0, N, int(nfft-no))
        signi_st = np.zeros((len(slices), k, k))
        for e, i in enumerate(slices):
            if i+nfft >= N:
                if trls:
                    datcut = np.concatenate((data[:, i:i+nfft], np.zeros((k, i+nfft-N, trls))), axis=1)
                else:
                    datcut = np.concatenate((data[:, i:i+nfft], np.zeros((k, i+nfft-N))), axis=1)
            else:
                datcut = data[:, i:i+nfft]
            signi_st[e] = self.significance(datcut, method, order=order, resolution=resolution,
                                            Nrep=Nrep, alpha=alpha, verbose=verbose, **params)
        return signi_st

    def __calc_multitrial(self, data, method='yw', order=None, fs=1, resolution=None, **params):
        "Calc multitrial averaged estimator for :func:`ConnectAR.bootstrap`"
        trials = data.shape[0]
        chosen = np.random.randint(trials, size=trials)
        ar, vr = Mvar().fit(data[:, :, chosen], order, method)
        rescalc = self.calculate(ar, vr, fs, resolution)
        return rescalc

    def significance(self, data, method, order=None, resolution=None, Nrep=10, alpha=0.05, verbose=True, **params):
        """
        Significance of connectivity estimators. It base on
        bootstrap :func:`ConnectAR.bootstrap` for multitrial case and
        surrogate data :func:`ConnectAR.surrogate` for one trial.
        Args:
          *data* : numpy.array
              data matrix
          *method* = 'yw': str
            method of MVAR parametersestimation
            all avaiable methods you can find in *fitting_algorithms*
          *order* = None : int
            MVAR model order, if None, it's chosen using default criterion
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
          *resolution* = None : int
              resolution (if None, it's 100 points)
          *verbose* = True : bool
            if True it prints dot on every realization, if False it's
            quiet.
          *params* :
              additional parameters specific for chosen estimator
        Returns:
          *signi_st* : numpy.array
              significance values, check :func:`Connect.levels`
        """
        if data.ndim > 2:
            signific = self.bootstrap(data, method, order=order, resolution=resolution,
                                      Nrep=Nrep, alpha=alpha, verbose=verbose, **params)
        else:
            signific = self.surrogate(data, method, order=order, resolution=resolution,
                                      Nrep=Nrep, alpha=alpha, verbose=verbose, **params)
        return signific

    def bootstrap(self, data, method, order=None, Nrep=10, alpha=0.05, fs=1, verbose=True, **params):
        """
        Bootstrap - random sampling with replacement of trials for *ConnectAR*.
        Args:
          *data* : numpy.array
              multichannel data matrix
          *method* : str
            method of MVAR parametersestimation
            all avaiable methods you can find in *fitting_algorithms*
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
          *order* = None : int
            MVAR model order, if None, it's chosen using default criterion
          *verbose* = True : bool
            if True it prints dot on every realization, if False it's
            quiet.
          *params* :
              additional parameters specific for chosen estimator
        Returns:
          *levelsigni* : numpy.array
              significance values, check :func:`Connect.levels`
        """
        resolution = 100
        if 'resolution' in params and params['resolution']:
            resolution = params['resolution']
        for i in range(Nrep):
            if verbose:
                print('.', end=' ')
            if i == 0:
                tmpsig = self.__calc_multitrial(data, method, order, fs, resolution)
                fres, k, k = tmpsig.shape
                signi = np.zeros((Nrep, fres, k, k))
                signi[i] = tmpsig
            else:
                signi[i] = self.__calc_multitrial(data, method, order, fs, resolution)
        if verbose:
            print('|')
        return self.levels(signi, alpha, k)

    def surrogate(self, data, method, Nrep=10, alpha=0.05, order=None, fs=1, verbose=True, **params):
        """
        Surrogate data testing for *ConnectAR* . Mixing data points in each channel.
        Significance level in calculated over all *Nrep* surrogate sets.
        Args:
          *data* : numpy.array
              multichannel data matrix
          *method* : str
            method of MVAR parameters estimation
            all avaiable methods you can find in *fitting_algorithms*
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
          *order* = None : int
            MVAR model order, if None, it's chosen using default criterion
          *verbose* = True : bool
            if True it prints dot on every realization, if False it's
            quiet.
          *params* :
              additional parameters specific for chosen estimator
        Returns:
          *levelsigni* : numpy.array
              significance values, check :func:`Connect.levels`
        """
        shdata = data.copy()
        k, N = data.shape
        resolution = 100
        if 'resolution' in params and params['resolution']:
            resolution = params['resolution']
        for i in range(Nrep):
            if verbose:
                print('.', end=' ')
            list(map(np.random.shuffle, shdata))
            ar, vr = Mvar().fit(shdata, order, method)
            if i == 0:
                rtmp = self.calculate(ar, vr, fs, resolution)
                reskeeper = np.zeros((Nrep, rtmp.shape[0], k, k))
                reskeeper[i] = rtmp
                continue
            reskeeper[i] = self.calculate(ar, vr, fs, resolution)
        if verbose:
            print('|')
        return self.levels(reskeeper, alpha, k)

############################
# MVAR based methods:


def dtf_fun(Acoef, Vcoef, fs, resolution, generalized=False):
    """
    Directed Transfer Function estimation from MVAR parameters.
    Args:
      *Acoef* : numpy.array
          array of shape (k, k, p) where *k* is number of channels and
          *p* is a model order.
      *Vcoef* : numpy.array
          prediction error matrix (k, k)
      *fs* = 1 : int
          sampling rate
      *resolution* = 100 : int
          number of spectrum data points
      *generalized* = False : bool
          generalized version or not
    Returns:
      *DTF* : numpy.array
          matrix with estimation results (*resolution*, k, k)
    References:
    .. [1] M. Kaminski, K.J. Blinowska. A new method of the description
           of the information flow. Biol.Cybern. 65:203-210, (1991).
    """
    A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution=resolution)
    res, k, k = A_z.shape
    DTF = np.zeros((res, k, k))
    if generalized:
        sigma = np.diag(Vcoef)
    else:
        sigma = np.ones(k)
    for i in range(res):
        mH = sigma*np.dot(H_z[i], H_z[i].T.conj()).real
        DTF[i] = (np.sqrt(sigma)*np.abs(H_z[i]))/np.sqrt(np.diag(mH)).reshape((k, 1))
    return DTF


def pdc_fun(Acoef, Vcoef, fs, resolution, generalized=False):
    """
    Partial Directed Coherence estimation from MVAR parameters.
    Args:
      *Acoef* : numpy.array
          array of shape (k, k, p) where *k* is number of channels and
          *p* is a model order.
      *Vcoef* : numpy.array
          prediction error matrix (k, k)
      *fs* = 1 : int
          sampling rate
      *resolution* = 100 : int
          number of spectrum data points
      *generalized* = False : bool
          generalized version or not
    Returns:
      *PDC* : numpy.array
          matrix with estimation results (*resolution*, k, k)
    References:
    .. [1] Sameshima, K., Baccala, L. A., Partial directed
           coherence: a new concept in neural structure determination.,
           2001, Biol. Cybern. 84, 463–474.
    """
    A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution=resolution)
    res, k, k = A_z.shape
    PDC = np.zeros((res, k, k))
    sigma = np.diag(Vcoef)
    for i in range(res):
        mA = (1./sigma[:, None])*np.dot(A_z[i].T.conj(), A_z[i]).real
        PDC[i] = np.abs(A_z[i]/np.sqrt(sigma))/np.sqrt(np.diag(mA))
    return PDC


class PartialCoh(ConnectAR):
    """
    PartialCoh - class inherits from :class:`ConnectAR` and overloads
    :func:`Connect.calculate` method.
    """
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=None):
        """
        Partial Coherence estimation from MVAR parameters.
        Args:
          *Acoef* : numpy.array
              array of shape (k, k, p) where *k* is number of channels and
              *p* is a model order.
          *Vcoef* : numpy.array
              prediction error matrix (k, k)
          *fs* = 1 : int
              sampling rate
          *resolution* = 100 : int
              number of spectrum data points
          *generalized* = False : bool
              generalized version or not
        Returns:
          *PC* : numpy.array
              matrix with estimation results (*resolution*, k, k)
        References:
        .. [1] G. M. Jenkins, D. G. Watts. Spectral Analysis and its
               Applications. Holden-Day, USA, 1969
        """
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution=resolution)
        res, k, k = A_z.shape
        PC = np.zeros((res, k, k))
        before = np.ones((k, k))
        before[0::2, :] *= -1
        before[:, 0::2] *= -1
        for i in range(res):
            D_z = np.linalg.inv(S_z[i])
            dd = np.tile(np.diag(D_z), (k, 1))
            mD = (dd*dd.T).real
            PC[i] = -1*before*(np.abs(D_z)/np.sqrt(mD))
        return np.abs(PC)


class PDC(ConnectAR):
    """
    PDC - class inherits from :class:`ConnectAR` and overloads
    :func:`Connect.calculate` method.
    """
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=100):
        "More in :func:`pdc_fun`."
        return pdc_fun(Acoef, Vcoef, fs, resolution)


class gPDC(ConnectAR):
    """
    gPDC - class inherits from :class:`ConnectAR` and overloads
    :func:`Connect.calculate` method.
    """
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=100):
        "More in :func:`pdc_fun`"
        return pdc_fun(Acoef, Vcoef, fs, resolution, generalized=True)


class DTF(ConnectAR):
    """
    DTF - class inherits from :class:`ConnectAR` and overloads
    :func:`Connect.calculate` method.
    """
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=100):
        "More in :func:`dtf_fun`."
        return dtf_fun(Acoef, Vcoef, fs, resolution)


class gDTF(ConnectAR):
    """
    gDTF - class inherits from :class:`ConnectAR` and overloads
    :func:`Connect.calculate` method.
    """
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=100):
        "More in :func:`dtf_fun`."
        return dtf_fun(Acoef, Vcoef, fs, resolution, generalized=True)


class ffDTF(ConnectAR):
    """
    ffDTF - class inherits from :class:`ConnectAR` and overloads
    :func:`Connect.calculate` method.
    """
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=100):
        """
        full-frequency Directed Transfer Function estimation from MVAR
        parameters.
        Args:
          *Acoef* : numpy.array
              array of shape (k, k, p) where *k* is number of channels and
              *p* is a model order.
          *Vcoef* : numpy.array
              prediction error matrix (k, k)
          *fs* = 1 : int
              sampling rate
          *resolution* = 100 : int
              number of spectrum data points
          *generalized* = False : bool
              generalized version or not
        Returns:
          *ffDTF* : numpy.array
              matrix with estimation results (*resolution*, k, k)
        References:
        .. [1] Korzeniewska, A.et. all. Determination of information flow direction
               among brain structures by a modified directed transfer function (dDTF)
               method. J. Neurosci. Methods 125, 195–207 (2003).
        """
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution=resolution)
        res, k, k = A_z.shape
        mH = np.zeros((res, k, k))
        for i in range(res):
            mH[i] = np.abs(np.dot(H_z[i], H_z[i].T.conj()))
        mHsum = np.sum(mH, axis=0)
        ffDTF = np.zeros((res, k, k))
        for i in range(res):
            ffDTF[i] = (np.abs(H_z[i]).T/np.sqrt(np.diag(mHsum))).T
        return ffDTF


class dDTF(ConnectAR):
    """
    dDTF - class inherits from :class:`ConnectAR` and overloads
    :func:`Connect.calculate` method.
    """
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=100):
        """
        direct Directed Transfer Function estimation from MVAR
        parameters. dDTF is a DTF multiplied in each frequency by
        Patrial Coherence.
        Args:
          *Acoef* : numpy.array
              array of shape (k, k, p) where *k* is number of channels and
              *p* is a model order.
          *Vcoef* : numpy.array
              prediction error matrix (k, k)
          *fs* = 1 : int
              sampling rate
          *resolution* = 100 : int
              number of spectrum data points
          *generalized* = False : bool
              generalized version or not
        Returns:
          *dDTF* : numpy.array
              matrix with estimation results (*resolution*, k, k)
        References:
        .. [1] Korzeniewska, A.et. all. Determination of information flow direction
               among brain structures by a modified directed transfer function (dDTF)
               method. J. Neurosci. Methods 125, 195–207 (2003).
        """
        A_z, H_z, S_z = spectrum(Acoef, Vcoef, fs, resolution=resolution)
        res, k, k = A_z.shape
        mH = np.zeros((res, k, k))
        for i in range(res):
            mH[i] = np.abs(np.dot(H_z[i], H_z[i].T.conj()))
        mHsum = np.sum(mH, axis=0)
        dDTF = np.zeros((res, k, k))
        before = np.ones((k, k))
        before[0::2, :] *= -1
        before[:, 0::2] *= -1
        for i in range(res):
            D_z = np.linalg.inv(S_z[i])
            dd = np.tile(np.diag(D_z), (k, 1))
            mD = (dd*dd.T).real
            PC = np.abs(-1*before*(np.abs(D_z)/np.sqrt(mD)))
            dDTF[i] = PC*(np.abs(H_z[i]).T/np.sqrt(np.diag(mHsum))).T
        return dDTF


class iPDC(ConnectAR):
    """
    iPDC - class inherits from :class:`ConnectAR` and overloads
    :func:`Connect.calculate` method.
    """
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=100):
        """
        instantaneous Partial Directed Coherence  from MVAR
        parameters.
        Args:
          *Acoef* : numpy.array
              array of shape (k, k, p+1) where *k* is number of channels and
              *p* is a model order. It's zero lag case.
          *Vcoef* : numpy.array
              prediction error matrix (k, k)
          *fs* = 1 : int
              sampling rate
          *resolution* = 100 : int
              number of spectrum data points
          *generalized* = False : bool
              generalized version or not
        Returns:
          *iPDC* : numpy.array
              matrix with estimation results (*resolution*, k, k)
        References:
        .. [1] Erla, S. et all Multivariate Autoregressive Model with Instantaneous
               Effects to Improve Brain Connectivity Estimation.
               Int. J. Bioelectromagn. 11, 74–79 (2009).
        """
        B_z = spectrum_inst(Acoef, Vcoef, fs, resolution=resolution)
        res, k, k = B_z.shape
        PDC = np.zeros((res, k, k))
        for i in range(res):
            mB = np.dot(B_z[i].T.conj(), B_z[i]).real
            PDC[i] = np.abs(B_z[i])/np.sqrt(np.diag(mB))
        return PDC


class iDTF(ConnectAR):
    """
    iDTF - class inherits from :class:`ConnectAR` and overloads
    :func:`Connect.calculate` method.
    """
    def calculate(self, Acoef=None, Vcoef=None, fs=None, resolution=100):
        """
        instantaneous Partial Directed Coherence  from MVAR
        parameters.
        Args:
          *Acoef* : numpy.array
              array of shape (k, k, p+1) where *k* is number of channels and
              *p* is a model order. It's zero lag case.
          *Vcoef* : numpy.array
              prediction error matrix (k, k)
          *fs* = 1 : int
              sampling rate
          *resolution* = 100 : int
              number of spectrum data points
          *generalized* = False : bool
              generalized version or not
        Returns:
          *iPDC* : numpy.array
              matrix with estimation results (*resolution*, k, k)
        References:
        .. [1] Erla, S. et all, Multivariate Autoregressive Model with Instantaneous
               Effects to Improve Brain Connectivity Estimation.
               Int. J. Bioelectromagn. 11, 74–79 (2009).
        """
        B_z = spectrum_inst(Acoef, Vcoef, fs, resolution=resolution)
        res, k, k = B_z.shape
        DTF = np.zeros((res, k, k))
        for i in range(res):
            Hb_z = np.linalg.inv(B_z[i])
            mH = np.dot(Hb_z, Hb_z.T.conj()).real
            DTF[i] = np.abs(Hb_z)/np.sqrt(np.diag(mH)).reshape((k, 1))
        return DTF

############################
# Fourier Transform based methods:


class Coherency(Connect):
    """
    Coherency - class inherits from :class:`Connect` and overloads
    :func:`Connect.calculate` method and *values_range* attribute.
    """
    def __init__(self):
        self.values_range = [0, 1]

    def calculate(self, data, cnfft=None, cno=None, window=np.hanning, im=False):
        """
        Coherency calculation using FFT mehtod.
        Args:
          *data* : numpy.array
              array of shape (k, N) where *k* is number of channels and
              *N* is number of data points.
          *cnfft* = None : int
              number of data points in window; if None, it is N/5
          *cno* = 0 : int
              overlap; if None, it is N/10
          *window* = np.hanning : <function> generating window with 1 arg *n*
              window function
          *im* = False : bool
              if False it return absolute value, otherwise complex number
        Returns:
          *COH* : numpy.array
              matrix with estimation results (*resolution*, k, k)
        References:
        .. [1] M. B. Priestley Spectral Analysis and Time Series.
               Academic Press Inc. (London) LTD., 1981
        """
        assert cnfft>cno, "overlap must be smaller than window"
        k, N = data.shape
        if not cnfft:
            cnfft = int(N/5)
        if cno is None:
            cno = int(N/10)
        winarr = window(cnfft)
        slices = range(0, N, int(cnfft-cno))
        ftsliced = np.zeros((len(slices), k, int(cnfft/2)+1), complex)
        for e, i in enumerate(slices):
            if i+cnfft >= N:
                datzer = np.concatenate((data[:, i:i+cnfft],
                                         np.zeros((k, i+cnfft-N))), axis=1)
                ftsliced[e] = np.fft.rfft(datzer*winarr, axis=1)
            else:
                ftsliced[e] = np.fft.rfft(data[:, i:i+cnfft]*winarr, axis=1)
        ctop = np.zeros((len(slices), k, k, int(cnfft/2)+1), complex)
        cdown = np.zeros((len(slices), k, int(cnfft/2)+1))
        for i in range(len(slices)):
            c1 = ftsliced[i, :, :].reshape((k, 1, int(cnfft/2)+1))
            c2 = ftsliced[i, :, :].conj().reshape((1, k, int(cnfft/2)+1))
            ctop[i] = c1*c2
            cdown[i] = np.abs(ftsliced[i, :, :])**2
        cd1  = np.mean(cdown, axis=0).reshape((k, 1, int(cnfft/2)+1))
        cd2  = np.mean(cdown, axis=0).reshape((1, k, int(cnfft/2)+1))
        cdwn = cd1*cd2
        coh  = np.mean(ctop, axis=0)/np.sqrt(cdwn)
        if not im:
            coh = np.abs(coh)
        return coh.T


class PSI(Connect):
    """
    PSI - class inherits from :class:`Connect` and overloads
    :func:`Connect.calculate` method.
    """
    def __init__(self):
        super(PSI, self).__init__()
        self.two_sided = True

    def calculate(self, data, band_width=4, psinfft=None, psino=0, window=np.hanning):
        """
        Phase Slope Index calculation using FFT mehtod.
        Args:
          *data* : numpy.array
              array of shape (k, N) where *k* is number of channels and
              *N* is number of data points.
          *band_width* = 4 : int
              width of frequency band where PSI values are summed
          *psinfft* = None : int
              number of data points in window; if None, it is N/5
          *psino* = 0 : int
              overlap; if None, it is N/10
          *window* = np.hanning : <function> generating window with 1 arg *n*
              window function
        Returns:
          *COH* : numpy.array
              matrix with estimation results (*resolution*, k, k)
        References:
        .. [1] Nolte G. et all, Comparison of Granger Causality and
               Phase Slope Index. 267–276 (2009).
        """
        k, N = data.shape
        if not psinfft:
            psinfft = int(N/4)
        assert psinfft > psino, "overlap must be smaller than window"
        coh = Coherency()
        cohval = coh.calculate(data, cnfft=psinfft, cno=psino, window=window, im=True)
        fq_bands = np.arange(0, int(psinfft/2)+1, band_width)
        psi = np.zeros((len(fq_bands)-1, k, k))
        for f in range(len(fq_bands)-1):
            ctmp = cohval[fq_bands[f]:fq_bands[f+1], :, :]
            psi[f] = np.imag(np.sum(ctmp[:-1, :, :].conj()*ctmp[1:, :, :], axis=0))
        return psi


class GCI(Connect):
    """
    GCI - class inherits from :class:`Connect` and overloads
    :func:`Connect.calculate` method.
    """
    def __init__(self):
        super(GCI, self).__init__()
        self.two_sided = False

    def calculate(self, data, gcimethod='yw', gciorder=None):
        """
        Granger Causality Index calculation from MVAR model.
        Args:
          *data* : numpy.array
              array of shape (k, N) where *k* is number of channels and
              *N* is number of data points.
          *gcimethod* = 'yw' : int
              MVAR parameters estimation model
          *gciorder* = None : int
              model order, if None appropiate value is chosen basic
              on default criterion
        Returns:
          *gci* : numpy.array
              matrix with estimation results (*resolution*, k, k)
        References:
        .. [1] Nolte G. et all, Comparison of Granger Causality and
               Phase Slope Index. 267–276 (2009).
        """
        k, N = data.shape
        arfull, vrfull = Mvar().fit(data, gciorder, gcimethod)
        gcval = np.zeros((k, k))
        for i in range(k):
            arix = [j for j in range(k) if i != j]
            ar_i, vr_i = Mvar().fit(data[arix, :], gciorder, gcimethod)
            for e, c in enumerate(arix):
                gcval[c, i] = np.log(vrfull[i, i]/vr_i[e, e])
        return np.tile(gcval, (2, 1, 1))

conn_estim_dc = {'dtf':   DTF,
                 'pdc':   PDC,
                 'ipdc':  iPDC,
                 'psi':   PSI,
                 'ffdtf': ffDTF,
                 'ddtf':  dDTF,
                 'gdtf':  gDTF,
                 'gpdc':  gPDC,
                 'pcoh':  PartialCoh,
                 'coh':   Coherency,
                 'gci':   GCI, }
