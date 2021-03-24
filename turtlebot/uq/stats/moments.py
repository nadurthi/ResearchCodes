#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

import numpy as np
import numpy.matlib as npmt


def MeanCov(X, w):
    N, dim = X.shape
    W = npmt.repmat(w.reshape((N, 1)), 1, dim)
    m = np.sum(np.multiply(X, W), axis=0)
    E = X - npmt.repmat(m, N, 1)
    Cov = np.matmul(E.T, np.multiply(W, E))

    return (m, Cov)
