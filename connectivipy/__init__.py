# -*- coding: utf-8 -*-
#! /usr/bin/env python

from data import Data
from conn import ConnectAR, spectrum, spectrumft, DTF
from mvarmodel import Mvar

# delete it later:
import numpy as np 
import pylab as py
from mvar.fitting import mvar_gen, vieiramorf, nutallstrand

__version__ = '0.05'
