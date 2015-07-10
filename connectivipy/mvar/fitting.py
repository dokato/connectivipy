# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
from numpy.linalg import inv 

def mvar_gen(Acf, n, omit=500):
    """
    Generating data point from matrix *A* with MVAR coefficients.
    Args:
      *Acf* : numpy.array
          array of dimension kxkxp where *k* is number of channels and
          *p* is a model order.
      *n* : int
          number of data points.
    Returns:
      *y* : np.array
          kx(n-omit) data points
    """
    p, chn, chn = Acf.shape
    y = np.zeros((chn, n + omit))
    sigma = np.diag(np.ones(chn))
    mu = np.zeros(chn)
    for i in range(p,n+omit):
        eps = np.random.multivariate_normal(mu, sigma)
        for k in xrange(0,p):
            yt = y[:,i-k-1].reshape((chn,1))
            y[:,i] += np.squeeze(np.dot(Acf[k],yt))
        y[:,i] += eps
    return y[:,omit:]

def mvar_gen_inst(Acf, n, omit=500):
    """
    Generating data point from matrix *A* with MVAR coefficients but it
    takes into account also zerolag interactions. So here Acf[0] means
    instantenous interaction not as in *mvar_gen* one data point lagged.
    Args:
      *Acf* : numpy.array
          array of dimension kxkxp where *k* is number of channels and
          *p* is a model order.
      *n* : int
          number of data points.
    Returns:
      *y* : np.array
          kx(n-omit) data points
    """
    p, chn, chn = Acf.shape
    y = np.zeros((chn, n + omit))
    sigma = np.diag(np.ones(chn))
    mu = np.zeros(chn)
    for i in range(p,n+omit):
        eps = np.random.multivariate_normal(mu, sigma)
        for k in xrange(0,p):
            yt = y[:,i-k].reshape((chn,1))
            y[:,i] += np.squeeze(np.dot(Acf[k],yt))
        y[:,i] += eps
    return y[:,omit:]


def ncov(x, y = [], p = 0, norm = True):
    """
    New covariance.
    Args:
      *x* : numpy.array
          one dimensional data.
      *y*=[] : numpy.array
          one dimensional data. If not given the autocovariance of *x*
          will be calculated.
       *p*=0: int
          window shift of input data. It can be negative as well.
       *norm*=True: bool
          normalization - if True the result is divided by length of *x*,
          otherwise it is not. 
    Returns:
      *kv* : np.array
          covariance matrix

    """
    C,N = x.shape
    cov = np.zeros((C,C,abs(p)+1))
    if len(y)==0 : y = x
    if p >= 0:
        for r in range(p+1):
            cov[:,:,r] = np.dot(x[:,:N-r],y[:,r:].T)
    else:
        for r in range(abs(p)+1):
            idxs = np.arange(-r,x.shape[1]-r)
            zy = y.take(idxs,axis=1, mode='wrap')
            cov[:,:,r] = np.dot(x[:,:N-r],zy[:,:N-r].T)
    if norm:
        kv = cov/N
    else:
        kv = cov 
    if p==0:
        kv = np.squeeze(kv)
    return kv

def vieiramorf(y,pmax=1):
    assert pmax>0, "pmax > 0"
    M,N = y.shape 
    f,b = y.copy(),y.copy()
    pef = ncov(y,norm=False)
    peb = pef.copy()
    arf = np.zeros((pmax,M,M))
    arb = np.zeros((pmax,M,M))
    for k in range(0,pmax):
        D = ncov(f[:,k+1:N],b[:,0:N-k-1],norm=False)
        arf[k,:,:] = np.dot(D,np.linalg.inv(peb))
        arb[k,:,:] = np.dot(D.T,np.linalg.inv(pef))
        
        tmp = f[:,k+1:] - np.dot(b[:,:N-k-1].T,arf[k,:,:].T).T
        b[:,:N-k-1] = b[:,:N-k-1] - np.dot(f[:,k+1:].T,arb[k,:,:].T).T
        f[:,k+1:] = tmp

        for i in range(k):
            tmpp = arf[i]-np.dot(arf[k],arb[k-i-1])
            arb[k-i-1,:,:] = arb[k-i-1,:,:] -np.dot(arb[k,:,:],arf[i,:,:])
            arf[i,:,:] = tmpp

        peb = ncov(b[:,:N-k-1],norm=False)
        pef = ncov(f[:,k+1:],norm=False)
    return arf, pef/N

def nutallstrand(y,pmax=1):
    assert pmax>0, "pmax > 0"
    M,N = y.shape 
    f,b = y.copy(),y.copy()
    pef = ncov(y,norm=False)
    peb = pef.copy()
    arf = np.zeros((pmax,M,M))
    arb = np.zeros((pmax,M,M))
    for k in range(0,pmax):
        D = ncov(f[:,k+1:N],b[:,0:N-k-1],norm=False)
        arf[k,:,:] = 2*np.dot(D,np.linalg.inv(peb + pef))
        arb[k,:,:] = 2*np.dot(D.T,np.linalg.inv(pef + peb))
        
        tmp = f[:,k+1:] - np.dot(b[:,:N-k-1].T,arf[k,:,:].T).T
        b[:,:N-k-1] = b[:,:N-k-1] - np.dot(f[:,k+1:].T,arb[k,:,:].T).T
        f[:,k+1:] = tmp

        for i in range(k):
            tmpp = arf[i]-np.dot(arf[k],arb[k-i-1])
            arb[k-i-1,:,:] = arb[k-i-1,:,:] -np.dot(arb[k,:,:],arf[i,:,:])
            arf[i,:,:] = tmpp

        peb = ncov(b[:,:N-k-1],norm=False)
        pef = ncov(f[:,k+1:],norm=False)
    return arf, pef/N

def yulewalker(y,pmax=1):
    assert pmax>0, "pmax > 0"
    chn,n = y.shape
    rr_f = ncov(y, p = pmax)
    rr_b = ncov(y, p = -1*pmax)
    q = np.zeros((pmax*chn,pmax*chn))
    acof = np.empty((pmax,chn,chn))
    for p in range(pmax):
        q[p*chn:(p+1)*chn,:] = np.hstack([ rr_f[:,:,x].T if x>=0 else rr_b[:,:,abs(x)].T for x in xrange(-1*p,pmax-p)])
    req = np.vstack(rr_b[:,:,x].T for x in xrange(1,pmax+1))
    a_solved = np.linalg.solve(q,req)
    var = np.copy(rr_f[:,:,0])
    for p in range(pmax):
        acof[p] = a_solved[p*chn:(p+1)*chn,:].T
        var -= np.dot(acof[p],rr_b[:,:,p+1].T)
    return acof, var

