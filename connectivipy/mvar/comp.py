#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def ldl(A):
    """
    LDL decomposition
    implementation from en.wikipedia.org
    """
    n = A.shape[1]
    L = np.eye(n)
    D = np.zeros(n)
    for j in xrange(n):
        D[j] = A[j,j] - np.dot(L[j,:j]**2, D[:j])
        for i in xrange(j+1,n):
            L[i,j] = (A[i, j] - np.dot(L[i,:j]*L[j,:j], D[:j]))/D[j]
    D = np.diag(D)
    return L, D, L.T


if __name__=='__main__':
    # test from wikipedia data
    A = np.array([[4,12,-16],[12,37,-43],[-16,-43,98]],dtype=float)
    l,d,lt = ldl(A)
    print l
    print d
    print lt
