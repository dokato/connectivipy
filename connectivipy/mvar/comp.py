# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from six.moves import range


def ldl(A):
    """
    LDL decomposition (implementation from *en.wikipedia.org/wiki/Cholesky_decomposition*)
    Args:
      *A* : numpy.array
          matrix kXk
    Returns:
      *L*, *D*, *LT* : np.array
          *L* is a lower unit triangular matrix, *D* is a diagonal matrix
          and *LT* is a transpose of *L*.
    """
    n = A.shape[1]
    L = np.eye(n)
    D = np.zeros(n)
    for j in range(n):
        D[j] = A[j, j] - np.dot(L[j, :j]**2, D[:j])
        for i in range(j+1, n):
            L[i, j] = (A[i, j] - np.dot(L[i, :j]*L[j, :j], D[:j]))/D[j]
    D = np.diag(D)
    return L, D, L.T


if __name__ == '__main__':
    # test of wikipedia data
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)
    l, d, lt = ldl(A)
    print(l)
    print(d)
    print(lt)
