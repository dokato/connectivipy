# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as si

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
    def __init__(self, data, fs, chan_names=[], data_info=''):
        dt_type = type(data)
        if dt_type == np.ndarray:
            self.data = data
        elif dt_type == str:
            self.data = self._load_file(data,data.split('.')[-1], data_info)
        self.fs = fs
        if self.data.shape[0]==len(chan_names):
            self.channames = chan_names

    def _load_file(self, dt_name, dt_type, data_info):
        '''
        Data loader.

        Args:
          *dt_name* : str
              path to file with appropieate format
          *dt_type* : str
              file extension (mat,)
          *dt_type* = '' : str
              additional file with data settings if needed
        Returns:
          *data* : np.array
        '''

        if dt_type == 'mat':
            mat_dict = si.loadmat(dt_name)
            data = [v for v in mat_dict.itervalues() if type(v) == np.ndarray][0]
        
        return data
            
    def filter(self,ftype):
        pass

    def resample(self, fs_new):
        '''
        Signal resample.
        
        Args:
          *fs_new*  new sampling frequency
        Returns:
          *frame* : np.array
             frame from camera
        '''
        pass 
    
    def estimate(self):
        pass

    def fit_mvar(self):
        pass
        
    def estimate(self):
        pass
    
    def plot_data(self):
        pass

    def plot_conn(self):
        pass

