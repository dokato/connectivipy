# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from mvar.fitting import *
 
class Mvar(object):
    
    __coefficients = None
    
    def fit_ns(self, data, order = 1):
        "Fit AR coef using Nuttall-Strand"
        pass

    def fit_ns(self, data, order = 1):
        "Fit AR coef using Vieira-Morf"
        pass

    def fit_yw(self, data, order = 1):
        "Fit AR coef using Yule-Walker"
        pass

    def order_akaike(self, data, ord_max = None):
        "Order"
        pass

    def order_fpc(self, data, ord_max = None):
        "Order"
        pass
    
    @property
    def coefficients(self):
        return self.__coefficients

    @property
    def coef(self):
        return self.__coefficients

