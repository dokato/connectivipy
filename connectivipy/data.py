# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, data, fs=1., chan_names=[], data_info=''):
        self.__data = self._load_file(data, data_info)
        self.__fs = fs
        if self.__data.shape[0]==len(chan_names):
            self.__channames = chan_names
        else:
            self.__channames = []
        self._parameters = {}
        self._parameters["mvar"] = False

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
            self.__multitrial = data.shape[2]
        else:
            self.__multitrial = False
        self.__chans = data.shape[0]
        self.__length = data.shape[1]
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
            k = self.__chans
            self.__Ar = np.zeros((self.__multitrial, p, k, k))
            self.__Vr = np.zeros((self.__multitrial, k, k))
            for r in xrange(self.__data.shape[2]):
                atmp, vtmp = Mvar().fit(self.__data[:,:,r], p, method)
                self.__Ar[r] = atmp
                self.__Vr[r] = vtmp
        self._parameters["mvar"] = True
        self._parameters["p"] = p
        self._parameters["mvarmethod"] = method

    def conn(self, method, **params):
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
            if isinstance(connobj,ConnectAR):
                self.__estim = connobj.calculate(self.__Ar,self.__Vr, self.__fs, **params)
            else:
                self.__estim = connobj.calculate(self.__data, **params)
        else:
            for r in xrange(self.__multitrial):
                if r==0:
                    if isinstance(connobj,ConnectAR):
                        self.__estim = connobj.calculate(self.__Ar[r], self.__Vr[r], self.__fs, **params)
                    else:
                        self.__estim = connobj.calculate(self.__data[:,:,r], **params)
                    continue
                if isinstance(connobj,ConnectAR):
                    self.__estim += connobj.calculate(self.__Ar[r], self.__Vr[r], self.__fs, **params)
                else:
                    self.__estim += connobj.calculate(self.__data[:,:,r], **params)
            self.__estim = self.__estim/self.__multitrial

        self._parameters["method"] = method
        self._parameters.update(params)
        return self.__estim

    def short_time_conn(self, method, nfft=None, no=None,**params):
        '''
        SHort-time connectivity.
        
        Args:
          *p* = None : int
            estimation order, default None
          *method* = 'yw' : str {'yw', 'ns', 'vm'}
            method of estimation, for full list please type:
            connectivipy.mvar_methods
        '''
        connobj = conn_estim_dc[method]()
        if not self._parameters.has_key("resolution"):
            self._parameters["resolution"] = None
        if isinstance(connobj,ConnectAR):
            self.__shtimest = connobj.short_time(self.__data, nfft=None, no=None,\
                                                 fs=self.__fs, order=self._parameters["p"],\
                                                 resol=self._parameters["resolution"])
        else:
            self.__shtimest = connobj.short_time(self.__data, **params)

        self._parameters["shorttime"] = method
        return self.__shtimest

    def significance(self, Nrep=100, alpha=0.05, **params):
        connobj = conn_estim_dc[self._parameters["method"]]()
        if not self.__multitrial:
            if isinstance(connobj,ConnectAR):
                self.__signific = connobj.surrogate(self.__data, Nrep=Nrep, alpha=alpha, 
                                                    method=self._parameters["mvarmethod"],\
                                                    fs=self.__fs, order=self._parameters["p"], **params)
            else:
                self.__signific = connobj.surrogate(self.__data, Nrep=Nrep,\
                                                    alpha=alpha, **params)
        else:
            if isinstance(connobj,ConnectAR):
                self.__signific = connobj.bootstrap(self.__Ar, self.__Vr, Nrep=Nrep,
                                                    alpha=alpha, fs=self.__fs, **params)
            else:
                self.__signific = connobj.bootstrap(self.__data, Nrep=Nrep,\
                                                    alpha=alpha, **params)
        return self.__signific

    def plot_data(self, trial=False, show=True):
        time = np.arange(0,self.__length)*1./self.__fs
        if self.__multitrial and not trial:
            plotdata = np.mean(self.__data, axis=2)
        elif self.__multitrial and trial:
            plotdata = self.__data[:,:,trial]
        else:
            plotdata = self.__data
        fig, axes = plt.subplots(self.__chans, 1)
        for i in xrange(self.__chans):
            axes[i].plot(time, plotdata[i,:], 'g')
            if self.__channames:
                axes[i].set_title(self.__channames[i])
        if show:
            plt.show()

    def plot_conn(self, name='', ylim=[0,1], xlim=None, signi=True, show=True):
        assert hasattr(self,'_Data__estim')==True, "No valid data!, Use calculation method first."
        fig, axes = plt.subplots(self.__chans, self.__chans)
        freqs = np.linspace(0, self.__fs//2, self.__estim.shape[0])
        if not xlim:
            xlim = [0, np.max(freqs)]
        if signi and hasattr(self,'_Data__signific'):
            flag_sig = True
        else:
            flag_sig = False
        for i in xrange(self.__chans):
            for j in xrange(self.__chans):
                if self.__channames and i==0:
                    axes[i, j].set_title(self.__channames[j]+" >", fontsize=12)
                if self.__channames and j==0:
                    axes[i, j].set_ylabel(self.__channames[i])
                axes[i, j].fill_between(freqs, self.__estim[:, i, j], 0)
                if flag_sig:
                    l = axes[i, j].axhline(y=self.__signific[i,j], color='r')
                axes[i, j].set_xlim(xlim)
                axes[i, j].set_ylim(ylim)
        plt.suptitle(name)
        plt.tight_layout()
        if show:
            plt.show()

    def plot_short_time_conn(self, name='',show=True):
        assert hasattr(self,'_Data__shtimest')==True, "No valid data! Use calculation method first."
        fig, axes = plt.subplots(self.__chans, self.__chans)
        freqs = np.linspace(0, self.__fs//2, self.__shtimest.shape[1])
        time = np.linspace(0, self.__length/self.__fs, self.__shtimest.shape[0])
        ticks_time = [0, self.__fs//2]
        ticks_freqs = [0, self.__length//self.__fs]
        dtmax = np.max(self.__shtimest)
        dtmin = np.min(self.__shtimest)
        plt.autoscale(False)
        for i in xrange(self.__chans):
            for j in xrange(self.__chans):
                if self.__channames and i==0:
                    axes[i, j].set_title(self.__channames[j]+" >", fontsize=12)
                if self.__channames and j==0:
                    axes[i, j].set_ylabel(self.__channames[i])
                axes[i, j].imshow(self.__shtimest[:,:,i,j].T, aspect='auto',\
                                  interpolation='none', origin='lower', vmin=dtmin, vmax=dtmax)
               # axes[i, j].set_yticks(ticks_time)
               # axes[i, j].set_xticks(ticks_freqs)
        plt.suptitle(name)
        plt.tight_layout()
        if show:
            plt.show()
    
    # accessors:
    @property
    def mvar_coefficients(self):
        if hasattr(self,'_Data__Ar') and hasattr(self,'_Data__Vr'):
            return (self.__Ar, self.__Vr)
        else:
            return (None, None)

    @property
    def mvarcoef(self):
        return self.mvar_coefficients

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
