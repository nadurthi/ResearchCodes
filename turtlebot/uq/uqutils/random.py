#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""


import numpy as np
import scipy.linalg as sclg
from numpy import linalg as LA

def genRandomCov(n,meanP=None,sigP=None):
    if sigP is None:
        meanP = np.zeros((n,n))
        sigP = np.eye(n)

    P = sclg.sqrtm(meanP) + np.matmul(sigP ,np.random.randn(n,n))
    return np.matmul(P,P.T)

def genRandomMeanCov(m,P,sigm,sigP):
    n=len(m)
    m = m + sigm*np.random.randn(n)
    P = genRandomCov(n,meanP=P,sigP=sigP)

    return (m,P)