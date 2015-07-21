# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as si
import scipy.signal as ss
from load.loaders import signalml_loader
from mvarmodel import Mvar
from conn import *

class Data(object):
    '''
    Class governing the communication between data array and 
    connectivity estimators.

    Args:
      *data* : numpy.array or str
          * array with data (kXN, k - channels nr, N - data points)
          * str - path to file with appropieate format
      *fs* : int
          sampling frequency
      *chan_names*: list
          names of channels
    '''
    def __init__(self, data, fs = 256., chan_names=[], data_info=''):
        self.__data = self._load_file(data, data_info)
        self.__fs = fs
        if self.__data.shape[0]==len(chan_names):
            self.__channames = chan_names

    def _load_file(self, data_what, data_info):
        '''
        Data loader.

        Args:
          *data_what* : str/numpy.array
              path to file with appropieate format or numpy data array
          *dt_type* : str
              file extension (mat,)
          *dt_type* = '' : str
              additional file with data settings if needed
        Returns:
          *data* : np.array
        '''
        dt_type = type(data_what)
        if dt_type == np.ndarray:
            data = data_what
        elif dt_type == str:
            if dt_type == 'mat':
                mat_dict = si.loadmat(data_what)
                if data_info:
                    key = data_info
                else:
                    key = data.split('.')[0]
                data = mat_dict[key]
            if data_info=='sml':
                data, sml = signalml_loader(data_what.split('.')[0])
                self.smldict = sml # here SignalML data is stored
        else:
            return False
        if len(data.shape)>2:
            self.__multitrial = True
        else:
            self.__multitrial = False
        return data
            
    def filter(self, b, a):
        '''
        Filter each channel of data using forward-backward  filter 
        *filtfilt* from *scipy.signal*.

        Args:
          *b,a* : np.array 
            Numerator *b*  / denominator *a* polynomials of the IIR filter.
        '''
        
        self.__data = ss.filtfilt(b,a,self.__data)

    def resample(self, fs_new):
        '''
        Signal resampling to new sampling frequency *new_fs* using
        *resample* function from *scipy.signal* (basing on Fourier method).
        
        Args:
          *fs_new* : int
            new sampling frequency
        '''
        new_nr_samples = (len(self.__data[0])*1./self.__fs)*fs_new
        self.__data = ss.resample(self.__data, new_nr_samples, axis=1)
        self.__fs = fs_new
    
    def estimate(self):
        pass

    def fit_mvar(self, p = None, method = 'yw'):
        '''
        Fitting MVAR coefficients.
        
        Args:
          *p* = None : int
            estimation order, default None
          *method* = 'yw' : str {'yw', 'ns', 'vm'}
            method of estimation, for full list please type:
            connectivipy.mvar_methods
        '''
        if not self.__multitrial:
            self.__Ar, self.__Vr = Mvar().fit(self.__data, p, method)
        else:
            k, N, tr = self.__data.shape
            self.__Ar = np.zeros((tr, p, k, k))
            self.__Vr = np.zeros((tr, k, k))
            for r in xrange(self.__data.shape[2]):
                atmp, vtmp = Mvar().fit(self.__data[:,:,r], p, method)
                self.__Ar[r] = atmp
                self.__Vr[r] = vtmp

    def conn(self, method = 'dtf', **params):
        '''
        Estimate connectivity pattern.
        
        Args:
          *p* = None : int
            estimation order, default None
          *method* = 'yw' : str {'yw', 'ns', 'vm'}
            method of estimation, for full list please type:
            connectivipy.mvar_methods
        '''
        connobj = conn_estim_dc[method]()
        if not self.__multitrial:
            self.__estim = connobj.calculate(self.__data, **params)
        else:
            k, N, tr = self.__data.shape
            for r in xrange(self.__data.shape[2]):
                self.__estim += connobj.calculate(self.__data[:,:,r], **params)
            self.__estim = self.__estim/tr
        return self.__estim
    
    def plot_data(self):
        pass

    def plot_conn(self):
        pass
    
    # accessors:
    @property
    def mvar_coefficients(self):
        if hasattr(self,'Ar') and hasattr(self,'Vr'):
            return (self.__Ar, self.__Vr)
        else:
            return None

    @property
    def mvarcoef(self):
        return self.mvar_coefficients()

    @property
    def data(self):
        return self.__data

    @property
    def fs(self):
        return self.__fs

    @property
    def srate(self):
        return self.__fs

    @property
    def channelnames(self):
        return self.__channames
