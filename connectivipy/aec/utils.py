import numpy as np
import scipy.signal as ss

FQ_BANDS = {'theta': [6, 7],
            'alpha': [8, 13],
            'beta': [15, 25],
            'low-gamma': [30, 45],
            'high-gamma': [55, 70]}

def check_bands_correct(band):
    return band in FQ_BANDS.keys()

def design_band_filter(lowcut, highcut, fs, rp = None, rs = None,
                       filttype = 'butter', btype = 'bandpass', order = 5):
    btypes = {'bandpass', 'bandstop'}
    filttypes = {'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'}
    if not btype in btypes:
        raise ValueError("This is only for band filters: {'bandpass', 'bandstop'}")
    if not filttype in filttypes:
        raise ValueError('Not supported filter type, check docs.')
    filtstr = 'ss.' + filttype + '(order,'
    if filttype == 'cheby1' or filttype == 'ellip':
        filtstr += 'rp,'
    if filttype == 'cheby2' or filttype == 'ellip':
        filtstr += 'rs,'
    filtstr += '[low, high], btype = btype)'
    f_nq = fs / 2
    low, high = lowcut / f_nq, highcut / f_nq
    b, a = eval(filtstr)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order = 4):
    return design_band_filter(lowcut, highcut, fs, order = order, btype = 'bandpass')

def butter_bandstop(lowcut, highcut, fs, order = 4):
    return design_band_filter(lowcut, highcut, fs, order = order, btype = 'bandstop')

def filter_band(data, fs, band = None, filter = None, filtfilt = True):
    if band == filter == None:
        raise ValueError("When *band* is None, *filter* can't be None")
    if filter is None:
        b, a = butter_bandpass(band[0], band[1], fs)
    else:
        b, a = filter
    if filtfilt:
        fdata = ss.filtfilt(b, a, data)
    else:
        fdata = ss.lfilt(b, a, data)
    return fdata

def calc_ampenv(data):
    return np.abs(ss.hilbert(data))
    