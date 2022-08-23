# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as sclg
from numpy import linalg as nplinalg
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
import numpy.matlib as npmt


def MeanCov(X, w):
    N, dim = X.shape
    W = npmt.repmat(w.reshape((N, 1)), 1, dim)
    m = np.sum(np.multiply(X, W), axis=0)
    E = X - npmt.repmat(m, N, 1)
    Cov = np.matmul(E.T, np.multiply(W, E))

    return (m, Cov)



def UT_sigmapoints(m, P):

    n = len(m)
    # % m=2 is 2n+1 points
    if n < 4:
        k = 3 - n
    else:
        k = 1

    x = np.zeros((2 * n + 1, n))
    w = np.zeros(2 * n + 1)
    x[0, :] = m
    w[0] = k / (n + k)

    A = sclg.sqrtm((n + k) * P)

    for i in range(n):
        x[i + 1, :] = (m + A[:, i])
        x[i + n + 1, :] = (m - A[:, i])
        w[i + 1] = 1 / (2 * (n + k))
        w[i + n + 1] = 1 / (2 * (n + k))

    return (x, w)


def propagateUKF(tk, tf,xfk, Pfk, propagateFunc,Qk,mu):
        
        Xk, Wk = UT_sigmapoints(xfk, Pfk)
        Xk1 = np.zeros(Xk.shape)
        for i in range(len(Wk)):  #FnG(t0, tf, x0, mu, tol = 1e-12)
            Xk1[i, :] = propagateFunc(tk, tf, Xk[i, :],mu)

        xfk1, Pfk1 = MeanCov(Xk1, Wk)

        Pfk1 = Pfk1 + Qk

        return xfk1, Pfk1

    

def measUpdateUKF(xfk, Pfk, sensorFunc, zk,Rk):
    """

    """
    hn=len(zk)
    
    X, W = UT_sigmapoints(xfk, Pfk)
    Z = np.zeros((X.shape[0], hn))

    for i in range(len(W)):
        Z[i, :] = sensorFunc(X[i, :])

    
    mz, Pz = MeanCov(Z, W)
    Pz = Pz + Rk

    Pxz = np.zeros((len(xfk), hn))
    for i in range(len(W)):
        Pxz = Pxz + W[i] * np.outer(X[i, :] - xfk, Z[i, :] - mz)


    K = np.matmul(Pxz, nplinalg.inv(Pz))
    
    if zk is None:
        xu = xfk
    else:
        xu = xfk + np.matmul(K, zk - mz)
        
    Pu = Pfk - multi_dot([K, Pz, K.T])

    return xu,Pu,mz,Pz

