# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from mvar.fitting import *
 
class Mvar(object):
    
    __coefficients = None

    def __init__(self):
        self._fit_dict = { 'yw': self._fit_yw,
                           'ns': self._fit_ns,
                           'vm': self._fit_vm
                         }

    def _fit_ns(self, data, order = 1):
        "Fit AR coef using Nuttall-Strand"
        return nutallstrand(data,order)

    def _fit_vm(self, data, order = 1):
        "Fit AR coef using Vieira-Morf"
        return vieiramorf(data,order)

    def _fit_yw(self, data, order = 1):
        "Fit AR coef using Yule-Walker"
        return yulewalker(data,order)

    def fit(self, data, order = None, method = 'yw'):
        #if order == None
        #   fit order using akaike etc...
        __coefficients = self._fit_dict[method](data,order)
        return __coefficients

    def _order_akaike(self, data, p_max = None, method = 'yw'):
        """
        Order akaike
        following Practical Biomedical Signal Analysis Using MATLAB eq 3.19
        """
        chn, N = data.shape 
        if p_max == None: 
            p_max = 20 # change to some criterion for max
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef,v_r) = self.fit(data,p+1,method)
            crit[p] = np.log(np.linalg.det(v_r))+2.*(p+1)*chn**2/N
        return np.argmin(crit), crit 

    def _order_hq(self, data, p_max = None, method = 'yw'):
        """
        Order Hannan-Quin
        following Practical Biomedical Signal Analysis Using MATLAB eq 3.20
        """
        chn, N = data.shape 
        if p_max == None:
            p_max = 20
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef,v_r) = self.fit(data,p+1,method)
            crit[p] = np.log(np.linalg.det(v_r))+2.*np.log(np.log(N))*(p+1)*chn**2/N
        return np.argmin(crit), crit 

    def _order_schwartz(self, data, p_max = None, method = 'yw'):
        """
        Order Schwartz
        following Practical Biomedical Signal Analysis Using MATLAB eq 3.21
        """
        chn, N = data.shape 
        if p_max == None:
            p_max = 20
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef,v_r) = self.fit(data,p+1,method)
            crit[p] = np.log(np.linalg.det(v_r))+np.log(N)*(p+1)*chn**2/N
        return np.argmin(crit)+1, crit 

    @property
    def coefficients(self):
        return self.__coefficients

    @property
    def coef(self):
        return self.__coefficients
