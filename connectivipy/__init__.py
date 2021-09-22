# -*- coding: utf-8 -*-

from .data import Data
from .conn import conn_estim_dc
from .mvarmodel import Mvar
from .mvar.fitting import mvar_gen, mvar_gen_inst, fitting_algorithms
from .plot import plot_conn

__all__ = ['data', 'conn']

__version__ = '0.3.7'
__version_vector__ = (0, 3, 7)
