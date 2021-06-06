import numpy as np
import numpy.linalg as nplalg
from numpy.linalg import multi_dot
import scipy.linalg as sclalg

def gaussianPDF1D(X,m,var):
    Pinv = 1/var
    a = X-m
    c = 1/np.sqrt(2*np.pi*var)
    y = c*np.exp(-0.5*Pinv*a**2)
    return y

def gaussianPDF(X,m,P):
    Pinv = nplalg.inv(P)
    a = X-m
    c = 1/np.sqrt(nplalg.det(2*np.pi*P))
    if X.ndim==1:
        y = c*np.exp(-0.5*multi_dot([a,Pinv,a]))
    else:
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = np.exp(-0.5*multi_dot([a[i],Pinv,a[i]]))
        y = c*y
    return y

def gaussianPDF_batchmean(x,M,p):
    Pinv = nplalg.inv(p)
    a = x-M
    c = 1/np.sqrt(nplalg.det(2*np.pi*p))

    y = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        y[i] = np.exp(-0.5*multi_dot([a[i],Pinv,a[i]]))
    y = c*y
    return y

def gaussianPDF_propratio(X,m,P,sig=1):
    """
    compute the ratio of p(X)/p(sig)
    this will tell us if the point is inside or outside the sig ellipsoid
    """
    dim = len(m)
    Pinv = nplalg.inv(P)
    a = X-m
    c = 1/np.sqrt(nplalg.det(2*np.pi*P))
    if X.ndim==1:
        y = c*np.exp(-0.5*multi_dot([a,Pinv,a]))
    else:
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = np.exp(-0.5*multi_dot([a[i],Pinv,a[i]]))
        y = c*y
    
    v= np.hstack([sig,np.zeros(dim-1)])
    vp = sclalg.sqrtm(P).dot(v)+m
    a = vp-m
    p =  c*np.exp(-0.5*multi_dot([a,Pinv,a]))
    
    return y/p

class GaussianPDF:
    def __init__(self,m,P):
        self.mean = m
        self.cov = P
    def pdf(self,x):
        Pinv = nplalg.inv(self.cov)
        a = x-self.mean
        return 1/np.sqrt(nplalg.det(2*np.pi*self.cov))*np.exp(-0.5*multi_dot([a,Pinv,a]))
