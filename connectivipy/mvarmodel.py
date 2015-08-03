# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from mvar.fitting import *
 
class Mvar(object):
    """
    Static class Mvar to multivariete autoregressive model
    fitting. Possible methods are in *_fit_dict* where key is
    acronym of algorithm and value is a function from *mvar.fitting*.
    """
    
    _fit_dict = {'yw': yulewalker,
                 'ns': nutallstrand,
                 'vm': vieiramorf}

    @classmethod
    def fit(cls, data, order=None, method='yw'):
        """
        Mvar model fitting.
        Args:
          *data* : numpy.array
              array with data (kXN, k - channels nr, N - data points)
          *order*=None : int
              model order, when default None it estimates order using
              akaike order criteria.
          *method* = 'yw': str
              name of mvar fitting algorithm, default Yule-Walker
        Returns:
          *Av* : np.array
              model coefficients (kXkXorder)
          *Vf* : np.array
              reflection matrix (kXk)
        """
        if order == None:
            order, crit_val = cls._order_hq(data, p_max=None, method=method)        
        return cls._fit_dict[method](data,order)
    
    @classmethod
    def _order_akaike(cls, data, p_max=None, method='yw'):
        """
        Order akaike
        following Practical Biomedical Signal Analysis Using MATLAB eq 3.19
        """
        chn, N = data.shape
        if p_max == None:
            p_max = 5 # change to some criterion for max
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef, v_r) = cls.fit(data, p+1, method)
            crit[p] = N*np.log(np.linalg.det(v_r))+2.*((p+1)*chn*(1+chn))
        return np.argmin(crit)+1, crit 
    @classmethod
    def _order_hq(cls, data, p_max=None, method='yw'):
        """
        Order Hannan-Quin
        following Practical Biomedical Signal Analysis Using MATLAB eq 3.20
        """
        chn, N = data.shape 
        if p_max == None:
            p_max = 5
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef, v_r) = cls.fit(data, p+1, method)
            crit[p] = np.log(np.linalg.det(v_r))+2.*np.log(np.log(N))*(p+1)*chn**2/N
        return np.argmin(crit)+1, crit 
    @classmethod
    def _order_schwartz(cls, data, p_max=None, method='yw'):
        """
        Order Schwartz
        following Practical Biomedical Signal Analysis Using MATLAB eq 3.21
        """
        chn, N = data.shape 
        if p_max == None:
            p_max = 5
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef,v_r) = cls.fit(data,p+1,method)
            crit[p] = np.log(np.linalg.det(v_r))+np.log(N)*(p+1)*chn**2/N
        return np.argmin(crit)+1, crit
    @classmethod
    def _order_fpe(cls, data, p_max=None, method='yw'):
        """
        Order FPE
        following Practical Biomedical Signal Analysis Using MATLAB eq 3.21
        """
        chn, N = data.shape 
        if p_max == None:
            p_max = 5
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef,v_r) = cls.fit(data,p+1,method)
            crit[p] = np.linalg.det(v_r) + chn*np.log((N+chn*(p+1)+1)/(N-chn*(p+1)-1))
        return np.argmin(crit)+1, crit
