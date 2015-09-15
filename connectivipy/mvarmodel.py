# -*- coding: utf-8 -*-
#! /usr/bin/env python

from __future__ import absolute_import
import numpy as np
from .mvar.fitting import *
from six.moves import range


class Mvar(object):
    """
    Static class *Mvar* to multivariete autoregressive model
    fitting. Possible methods are in *fitting_algorithms* where key is
    acronym of algorithm and value is a function from *mvar.fitting*.
    """

    fit_dict = fitting_algorithms

    @classmethod
    def fit(cls, data, order=None, method='yw'):
        """
        Mvar model fitting.
        Args:
          *data* : numpy.array
              array with data shaped (k, N), k - channels nr,
              N-data points)
          *order* = None : int
              model order, when default None it estimates order using
              akaike order criteria.
          *method* = 'yw': str
              name of mvar fitting algorithm, default Yule-Walker
              all avaiable methods you can find in *fitting_algorithms*
        Returns:
          *Av* : numpy.array
              model coefficients (kXkXorder)
          *Vf* : numpy.array
              reflection matrix (kXk)
        """
        if order is None:
            order, crit_val = cls.order_hq(data, p_max=None, method=method)
        return cls.fit_dict[method](data, order)

    @classmethod
    def order_akaike(cls, data, p_max=None, method='yw'):
        """
        Akaike criterion of MVAR order estimation.

        Args:
          *data* : numpy.array
              multichannel data in shape (k, n) for one trial case and
              (k, n, tr) for multitrial
              k - nr of channels, n -data points, tr - nr of trials
          *p_max* = 5 : int
              maximal model order
          *method* = 'yw' : str
              name of the mvar calculation method
        Returns:
          *best_order* : int
              minimum of *crit* array
          *crit* : numpy.array
              order criterion values for each value of order *p*
              starting from 1
        References:
        .. [1] Blinowska K. J., Zygierewicz J., (2012) Practical
               biomedical signal analysis using MATLAB.
               Boca Raton: Taylor & Francis.
        """
        if data.ndim > 2:
            chn, N, _ = data.shape
        else:
            chn, N = data.shape
        if p_max is None:
            p_max = 5  # change to some criterion for max
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef, v_r) = cls.fit(data, p+1, method)
            crit[p] = N*np.log(np.linalg.det(v_r))+2.*((p+1)*chn*(1+chn))
        return np.argmin(crit)+1, crit

    @classmethod
    def order_hq(cls, data, p_max=None, method='yw'):
        """
        Hannan-Quin criterion of MVAR order estimation.

        Args:
          *data* : numpy.array
              multichannel data in shape (k, n) for one trial case and
              (k, n, tr) for multitrial
              k - nr of channels, n -data points, tr - nr of trials
          *p_max* = 5 : int
              maximal model order
          *method* = 'yw' : str
              name of the mvar calculation method
        Returns:
          *best_order* : int
              minimum of *crit* array
          *crit* : numpy.array
              order criterion values for each value of order *p*
              starting from 1
        References:
        .. [1] Blinowska K. J., Zygierewicz J., (2012) Practical
               biomedical signal analysis using MATLAB.
               Boca Raton: Taylor & Francis.
        """
        if data.ndim > 2:
            chn, N, _ = data.shape
        else:
            chn, N = data.shape
        if p_max is None:
            p_max = 5
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef, v_r) = cls.fit(data, p+1, method)
            crit[p] = np.log(np.linalg.det(v_r))+2.*np.log(np.log(N))*(p+1)*chn**2/N
        return np.argmin(crit)+1, crit

    @classmethod
    def order_schwartz(cls, data, p_max=None, method='yw'):
        """
        Schwartz criterion of MVAR order estimation.

        Args:
          *data* : numpy.array
              multichannel data in shape (k, n) for one trial case and
              (k, n, tr) for multitrial
              k - nr of channels, n -data points, tr - nr of trials
          *p_max* = 5 : int
              maximal model order
          *method* = 'yw' : str
              name of the mvar calculation method
        Returns:
          *best_order* : int
              minimum of *crit* array
          *crit* : numpy.array
              order criterion values for each value of order *p*
              starting from 1
        References:
        .. [1] Blinowska K. J., Zygierewicz J., (2012) Practical
               biomedical signal analysis using MATLAB.
               Boca Raton: Taylor & Francis.
        """
        if data.ndim > 2:
            chn, N, _ = data.shape
        else:
            chn, N = data.shape
        if p_max is None:
            p_max = 5
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef, v_r) = cls.fit(data, p+1, method)
            crit[p] = np.log(np.linalg.det(v_r))+np.log(N)*(p+1)*chn**2/N
        return np.argmin(crit)+1, crit

    @classmethod
    def order_fpe(cls, data, p_max=None, method='yw'):
        """
        Final Prediction Error criterion of MVAR order estimation.
        (not recommended)
        Args:
          *data* : numpy.array
              multichannel data in shape (k, n) for one trial case and
              (k, n, tr) for multitrial
              k - nr of channels, n -data points, tr - nr of trials
          *p_max* = 5 : int
              maximal model order
          *method* = 'yw' : str
              name of the mvar calculation method
        Returns:
          *best_order* : int
              minimum of *crit* array
          *crit* : numpy.array
              order criterion values for each value of order *p*
              starting from 1
        References:
        .. [1] Akaike H, (1970), Statistical predictor identification,
               Ann. Inst. Statist. Math., 22 203–217.
        """
        if data.ndim > 2:
            chn, N, _ = data.shape
        else:
            chn, N = data.shape
        if p_max is None:
            p_max = 5
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef, v_r) = cls.fit(data, p+1, method)
            crit[p] = np.linalg.det(v_r) + chn*np.log((N+chn*(p+1)+1)/(N-chn*(p+1)-1))
        return np.argmin(crit)+1, crit
