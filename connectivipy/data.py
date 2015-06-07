# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as si
from loaders import signalml_loader

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
                self.smldict = sml # here SignaML data is stored
        else:
            pass 
        return data
            
    def filter(self,ftype):
        pass

    def resample(self, fs_new):
        '''
        Signal resampling.
        
        Args:
          *fs_new*  new sampling frequency
        '''
        pass 
    
    def estimate(self):
        pass

    def fit_mvar(self):
        pass
    
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