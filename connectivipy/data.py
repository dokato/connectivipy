# -*- coding: utf-8 -*-
#! /usr/bin/env python

import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
          * array with data (kXNxR, k - channels nr, N - data points,
                             R - nr of trials)
          * str - path to file with appropieate format
      *fs* = 1: int
          sampling frequency
      *chan_names* = []: list
          names of channels
      *data_info* = '': string
          other information about the data
    '''
    def __init__(self, data, fs=1., chan_names=[], data_info=''):
        self.__data = self._load_file(data, data_info)
        self.__fs = fs
        if self.__data.shape[0]==len(chan_names):
            self.__channames = chan_names
        else:
            self.__channames = ["x"+str(i) for i in range(self.__chans)]
        self.data_info = data_info
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
        if self.__multitrial:
            for r in xrange(self.__multitrial):
                self.__data[:,:,r] = ss.filtfilt(b,a,self.__data[:,:,r])
        else:
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
    
    def fit_mvar(self, p=None, method='yw'):
        '''
        Fitting MVAR coefficients.
        
        Args:
          *p* = None : int
            estimation order, default None
          *method* = 'yw' : str {'yw', 'ns', 'vm'}
            method of estimation
        '''
        self.__Ar, self.__Vr = Mvar().fit(self.__data, p, method)
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
            method of estimation
        '''
        connobj = conn_estim_dc[method]()
        if isinstance(connobj,ConnectAR):
            self.__estim = connobj.calculate(self.__Ar,self.__Vr, self.__fs, **params)
        else:
            if not self.__multitrial:
                self.__estim = connobj.calculate(self.__data, **params)
            else:
                for r in xrange(self.__multitrial):
                    if r==0:
                        self.__estim = connobj.calculate(self.__data[:,:,r], **params)
                        continue
                    self.__estim += connobj.calculate(self.__data[:,:,r], **params)
                self.__estim = self.__estim/self.__multitrial
        self._parameters["method"] = method
        self._parameters["y_lim"] = connobj.values_range
        self._parameters.update(params)
        return self.__estim

    def short_time_conn(self, method, nfft=None, no=None, **params):
        '''
        Short-time connectivity.
        
        Args:
          *method* = 'yw' : str {'yw', 'ns', 'vm'}
            method of estimation, for full list please type:
            connectivipy.mvar_methods
          *nfft* = None : int
            number of data points in window; if None, it is signal length
            N/5.
          *no* = None : int
            number of data points in overlap; if None, it is signal length
            N/10.
          *params*
            other parameters for specific estimator
        '''
        connobj = conn_estim_dc[method]()
        self._parameters.update(params)
        arg = inspect.getargspec(connobj.calculate)
        newparams = self.__make_params_dict(arg[0])
        if not self._parameters.has_key("p"):
            if params.has_key("order"):
                self._parameters["p"] = params["order"]
            else:
                self._parameters["p"] = None
        if not nfft:
            nfft = int(self.__length/5)
        if not no:
            no = int(self.__length/10)
        if not self._parameters.has_key("resolution"):
            self._parameters["resolution"] = 100
        if isinstance(connobj,ConnectAR):
            self.__shtimest = connobj.short_time(self.__data, nfft=nfft, no=no,\
                                                 fs=self.__fs, order=self._parameters["p"],\
                                                 resol=self._parameters["resolution"])
        else:
            if self.__multitrial:
                for r in xrange(self.__multitrial):
                    if r==0:
                        self.__shtimest = connobj.short_time(self.__data[:,:,r], nfft=nfft, no=no, **newparams)
                        continue
                    self.__shtimest += connobj.short_time(self.__data[:,:,r], nfft=nfft, no=no, **newparams)
                self.__shtimest/=self.__multitrial
            else:
                self.__shtimest = connobj.short_time(self.__data, nfft=nfft, no=no, **newparams)
        self._parameters["shorttime"] = method
        self._parameters["nfft"] = nfft
        self._parameters["no"] = no
        return self.__shtimest

    def significance(self, Nrep=100, alpha=0.05, **params):
        '''
        Statistical significance values of connectivity estimation method.
        
        Args:
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
        Returns:
          *signi*: numpy.array
            matrix in shape of (k, k) with values for each pair of
            channels
        '''
        connobj = conn_estim_dc[self._parameters["method"]]()
        self._parameters.update(params)
        arg = inspect.getargspec(connobj.calculate)
        newparams = self.__make_params_dict(arg[0])
        if not self.__multitrial:
            if isinstance(connobj,ConnectAR):
                self.__signific = connobj.surrogate(self.__data, Nrep=Nrep, alpha=alpha, 
                                                    method=self._parameters["mvarmethod"],\
                                                    fs=self.__fs, order=self._parameters["p"], **newparams)
            else:
                self.__signific = connobj.surrogate(self.__data, Nrep=Nrep,\
                                                    alpha=alpha, **newparams)
        else:
            if isinstance(connobj,ConnectAR):
                self.__signific = connobj.bootstrap(self.__data, Nrep=Nrep, alpha=alpha, 
                                                    method=self._parameters["mvarmethod"],\
                                                    fs=self.__fs, order=self._parameters["p"], **newparams)
            else:
                self.__signific = connobj.bootstrap(self.__data, Nrep=Nrep,\
                                                    alpha=alpha, **newparams)
        return self.__signific
    
    def short_time_significance(self, Nrep=100, alpha=0.05, nfft=None, no=None, **params):
        '''
        Statistical significance values of short-time version of
        connectivity estimation method.
        
        Args:
          *Nrep* = 100 : int
            number of resamples
          *alpha* = 0.05 : float
            type I error rate (significance level)
          *nfft* = None : int
            number of data points in window; if None, it is taken from
            :func:`Data.short_time_conn` method.
          *no* = None : int
            number of data points in overlap; if None, it is taken from
            *short_time_conn* method.
        Returns:
          *signi*: numpy.array
            matrix in shape of (k, k) with values for each pair of
            channels
        '''
        if not nfft:
            nfft = self._parameters["nfft"]
        if not no:
            no = self._parameters["no"]
        connobj = conn_estim_dc[self._parameters["shorttime"]]()
        self._parameters.update(params)
        arg = inspect.getargspec(connobj.calculate)
        newparams = self.__make_params_dict(arg[0])
        if isinstance(connobj,ConnectAR):
            self.__st_signific = connobj.short_time_significance(self.__data, Nrep=Nrep, alpha=alpha, 
                                                              method=self._parameters["mvarmethod"],\
                                                              fs=self.__fs, order=self._parameters["p"],
                                                              nfft=nfft, no=no, **newparams)
        else:
            self.__st_signific = connobj.short_time_significance(self.__data, Nrep=Nrep,\
                                                              nfft=nfft, no=no,\
                                                              alpha=alpha, **newparams)
        return self.__st_signific

    def plot_data(self, trial=0, show=True):
        '''
        Plot data in a subplot for each channel.
        
        Args:
          *trial* = 0 : int
            if there is multichannel data it should be a number of trial
            you want to plot.
          *show* = True : boolean
            show the plot or not
        '''

        time = np.arange(0,self.__length)*1./self.__fs
        if self.__multitrial:
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

    def plot_conn(self, name='', ylim=None, xlim=None, signi=True, show=True):
        '''
        Plot connectivity estimation results.
        
        Args:
          *name* = '' : str
            title of the plot
          *ylim* = None : list
            range of y-axis values shown, e.g. [0,1]
            *None* means that default values of given estimator are taken
            into account
          *xlim* = None : list [from (int), to (int)]
            range of y-axis values shown, if None it is from 0 to Nyquist frequency
          *signi* = True : boolean
            if significance levels are calculated they are shown in the plot
          *show* = True : boolean
            show the plot or not            
        '''
        assert hasattr(self,'_Data__estim')==True, "No valid data!, Use calculation method first."
        fig, axes = plt.subplots(self.__chans, self.__chans)
        freqs = np.linspace(0, self.__fs//2, self.__estim.shape[0])
        if not xlim:
            xlim = [0, np.max(freqs)]
        if not ylim:
            ylim = self._parameters["y_lim"]
            if ylim[0] == None:
                ylim[0] = np.min(self.__estim)
            if ylim[1] == None:
                ylim[1] = np.max(self.__estim)
        two_sides = False
        if signi and hasattr(self,'_Data__signific'):
            flag_sig = True
            if len(self.__signific.shape)>2:
                two_sides = True
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
                    if two_sides:
                        l_u = axes[i, j].axhline(y=self.__signific[0,i,j], color='r')
                        l_d = axes[i, j].axhline(y=self.__signific[1,i,j], color='r')
                    else:
                        l = axes[i, j].axhline(y=self.__signific[i,j], color='r')
                axes[i, j].set_xlim(xlim)
                axes[i, j].set_ylim(ylim)
                if i!=self.__chans-1:
                    axes[i,j].get_xaxis().set_visible(False)
                if j!=0:
                    axes[i,j].get_yaxis().set_visible(False)
        plt.suptitle(name,y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        if show:
            plt.show()

    def plot_short_time_conn(self, name='', signi=True, percmax=1., show=True):
        '''
        Plot short-time version of estimation results.
        
        Args:
          *name* = '' : str
            title of the plot
          *signi* = True : boolean
            reset irrelevant values; it works only after short time
            significance calculation using *short_time_significance*
          *percmax* = 1. : float (0,1)
            percent of maximal value which is maximum on the color map
          *show* = True : boolean
            show the plot or not            
        '''
        assert hasattr(self,'_Data__shtimest')==True, "No valid data! Use calculation method first."
        shtvalues = self.__shtimest
        if signi and hasattr(self,'_Data__st_signific'):
            if len(self.__st_signific.shape)>3:
                shtvalues = self.fill_nans(shtvalues,self.__st_signific[:,0,:,:])
                shtvalues = self.fill_nans(shtvalues,self.__st_signific[:,1,:,:])
            else:
                shtvalues = self.fill_nans(shtvalues,self.__st_signific)
        fig, axes = plt.subplots(self.__chans, self.__chans)
        freqs = np.linspace(0, self.__fs//2, 4)
        time = np.linspace(0, self.__length/self.__fs, 5)
        ticks_time = [0, self.__fs//2]
        ticks_freqs = [0, self.__length//self.__fs]
        # mask diagonal values to not contaminate the plot
        mask = np.zeros(shtvalues.shape)
        for i in xrange(self.__chans):
            mask[:,:,i,i] = 1
        masked_shtimest = np.ma.array(shtvalues, mask=mask)
        dtmax = np.max(masked_shtimest)*percmax
        dtmin = np.min(masked_shtimest)
        for i in xrange(self.__chans):
            for j in xrange(self.__chans):
                if self.__channames and i==0:
                    axes[i, j].set_title(self.__channames[j]+" >", fontsize=12)
                if self.__channames and j==0:
                    axes[i, j].set_ylabel(self.__channames[i])
                img = axes[i, j].imshow(shtvalues[:,:,i,j].T, aspect='auto',\
                                       extent=[0,self.__length/self.__fs,0,self.__fs//2], \
                                       interpolation='none', origin='lower', vmin=dtmin, vmax=dtmax)
                if i!=self.__chans-1:
                    axes[i,j].get_xaxis().set_visible(False)
                if j!=0:
                    axes[i,j].get_yaxis().set_visible(False)
                xt  = np.array(axes[i,j].get_xticks())/self.__fs
        plt.suptitle(name,y=0.98)
        plt.tight_layout()
        fig.subplots_adjust(top=0.92, right=0.91)
        cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.7])
        cbar_ax.tick_params(labelsize=10)
        fig.colorbar(img, cax=cbar_ax)
        if show:
            plt.show()

    def export_trans3d(self, filename='conn_trnas3d.dat', freq_band=[]):
        '''
        Export connectivity data to trans3D data file in order to make
        3D arrow plots.
        Args:
          *filename* = 'conn_trnas3d.dat' : str
            title of the plot
          *freq_band* = [] : list
            frequency range [from_value, to_value] in Hz.            
        '''
        trdef_str = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -8, 0, 0, 0, 1]
        tr_str = [0.0051, 0.0074, -3.4523, 0., -2.7277, 2.1161, 0.0005, 0., 2.1161, 2.7276, 0., -8., 0., 0., 0., 1.]
        content = ";electrodes = " + " ".join(self.__channames)
        content += "\r\n;start = -0.500000\r\n"
        content += ";samplerate = 12\r\n"
        content += ";transform_default = " + " ".join([ str(x) for x in trdef_str[:self.__chans]]) + "\r\n"
        content += ";transform = " + " ".join([ str(x) for x in tr_str[:self.__chans]]) + "\r\n"
        content += "\r\n"
        # integrate value of estimator in given frequency band
        freqs = np.linspace(0, int(self.__fs/2), self.__estim.shape[0])
        if len(freq_band)==0:
            ind1 = 0
            ind2 = len(freqs)
        else:
            ind1 = np.where(freqs>=freq_band[0])[0][0]
            ind2 = np.where(freqs>=freq_band[1])[0][0]
        cnest = np.mean(self.__estim[ind1:ind2,:,:], axis=0)
        for i in xrange(self.__chans):
            content+= "  " + "\t".join([ str(x) for x in cnest[i]]) + "\r\n" 
        with open(filename,'wb') as fl:
            fl.write(content)
    
    # auxiliary methods:

    def __make_params_dict(self, args):
        """
        Making list of parameters from *self._parameters*
        Args:
          *args* : list
            list with parameters of *calculate* method of specific
            estimator
        Returns:
          *newparams* : dict
            dictionary with new parameters
        """
        newparams = {}
        for ag in args[1:]:
            if ag in ['data']:
                continue
            if self._parameters.has_key(ag):
                newparams[ag] = self._parameters[ag]
        return newparams

    def fill_nans(self, values, borders):
        '''
        Fill nans where *values* < *borders* (independent of frequency).
        
        Args:
          *values* : numpy.array
            array of shape (time, freqs, channels, channels) to fill nans
          *borders* : numpy.array
            array of shape (time, channels, channels) with limes
            values
        Returns:
          *values_nans* : numpy.array
            array of shape (time, freq, channels, channels) with nans
            where values were less than appropieate value from *borders*
        '''
        tm, fr, k, k = values.shape
        for i in xrange(fr):
            values[:,i,:,:][values[:,i,:,:]<borders] = 0
        return values
    
    # accessors:
    @property
    def mvar_coefficients(self):
        "Returns mvar coefficients if calculated"
        if hasattr(self,'_Data__Ar') and hasattr(self,'_Data__Vr'):
            return (self.__Ar, self.__Vr)
        else:
            return (None, None)

    @property
    def mvarcoef(self):
        "Returns mvar coefficients if calculated"
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
