# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from abc import ABCMeta, abstractmethod

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
