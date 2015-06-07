# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from mvar.fitting import *
 
class Mvar(object):
    
    __coefficients = None

    _fit_dict = { 'yw': self._fit_yw,
                 'ns': self._fit_ns,
                 'vm': self._fit_vm
                 }
    
    def _fit_ns(self, data, order = 1):
        "Fit AR coef using Nuttall-Strand"
        return nutallstrand(data,order)

    def _fit_vm(self, data, order = 1):
        "Fit AR coef using Vieira-Morf"
        return viermorf(data,order)

    def _fit_yw(self, data, order = 1):
        "Fit AR coef using Yule-Walker"
        return yulewalker(data,order)

    def fit(self, data, order = None, method = 'yw'):
        #if order == None
        #   fit order using akaike etc...
        try:
            __coefficients = _fit_dict[method](data,order)
        return __coefficients

    def _order_akaike(self, data, p_max = None, method = 'yw'):
        "Order akaike"
        chn, N = data.shape 
        if p_max == None:
            p_max = chn + 1
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef,v_r) = mult_AR(dat,p+1,meth_num)
            crit[p]=np.log(np.linalg.det(v_r))+2.*(p+1)*chn**2/N
        return np.argmin(crit), crit 

    def _order_fpc(self, data, ord_max = None, method = 'yw'):
        "Order fpc"
        chn, N = data.shape 
        if p_max == None:
            p_max = chn + 1
        crit = np.zeros(p_max)
        for p in range(p_max):
            (a_coef,v_r) = mult_AR(dat,p+1,meth_num)
            crit[p]= np.linalg.det(v_r)*((N+chn*(p+1))/(N-chn*(p+1)))**chn
        return np.argmin(crit), crit 
        
    @property
    def coefficients(self):
        return self.__coefficients

    @property
    def coef(self):
        return self.__coefficients
