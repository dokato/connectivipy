# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as si
import scipy.signal as ss
from load.loaders import signalml_loader
from mvarmodel import Mvar

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
            pass 
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

    def resample(self, new_fs):
        '''
        Signal resampling to new sampling frequency *new_fs* using
        *resample* function from *scipy.signal* (basing on Fourier method).
        
        Args:
          *new_fs* : int
            new sampling frequency
        '''
        new_nr_samples = (len(self.__data[0])*1./self.__fs)*fs_new
        self.__data = ss.resample(self.__data, new_nr_samples)
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
        mvar = Mvar()
    
    def plot_data(self):
        pass

    def plot_conn(self):
        pass
    
    # accessors:
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
