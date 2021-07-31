# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc
from numba import njit, prange,jit
import copy

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.neighbors import KDTree
# from utils.plotting import geometryshapes as utpltgmshp
import time
from scipy.optimize import minimize, least_squares
from scipy import interpolate
from fastdist import fastdist
from uq.gmm import gmmfuncs as uqgmmfnc
import pdb
import networkx as nx
from uq.gmm import gmmfuncs as uqgmmfnc
from scipy.optimize import least_squares
from uq.quadratures import cubatures as uqcub
import math
numba_cache=True
dtype=np.float64


#%%

@jit(float64(float64[:], float64[:,::1], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,cache=True) 
def getcostalign(x,Xt,MU,P,W):
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    p=uqgmmfnc.gmm_eval_fast(Xn,MU,P,W)+0.01
    logp = -np.log(p)
    f = np.mean(logp)
    # logp = np.diag(f(Xn[:,0],Xn[:,1]))
    return f



#%%

@jit(numba.types.Tuple((float64[:],float64))(float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def extractPosAngle(H):
    theta = np.arctan2(-H[0,1],H[0,0])
    txy = H[:2,2]
    return txy,theta

@jit(float64[:,:](float64, float64[:]),nopython=True, nogil=True,parallel=False,cache=True) 
def getHmat(th,t):
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    H=np.identity(3)
    H[0:2,0:2]=R
    H[0:2,2]=t
    return H

#%%
@jit(nopython=True, nogil=True,parallel=False,cache=True) #float64[:](float64[:,:], float64[:,:], float64[:,:,:],float64[:], float64[:,::1],int32[:,:]),
def gridsearch_alignment(Posegrid,MU,P,W,X, returnBest = 1):
    
    m=np.zeros(Posegrid.shape[0])
    for i in numba.prange(Posegrid.shape[0]):
        m[i]=getcostalign(Posegrid[i],X,MU,P,W)
    ind = np.argmin(m)
    res = Posegrid[ind]
    
    if returnBest == 1:
        return res #typically desired return
    else:
        return m  #Returns all poses rather than just the minimum, this is for brute force allignment


@jit(numba.types.Tuple((float64,float64[:]))(float64[:], float64[:,::1], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=True) 
def getcostgradient(x,Xt,MU,P,W):
    npt=Xt.shape[0]
    ncomp=MU.shape[0]
    
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    pcomp=uqgmmfnc.gmm_evalcomp_fast(Xn,MU,P,W)
    p = np.sum(pcomp,axis=1)+0.01
    logp = -np.log(p)
    # ind = logp<np.mean(logp)
    f = np.mean(logp)    
    
    invp = 1/p  # this is a vec with each element for one point in Xt
    dRdth = np.array([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])
    invPP = np.zeros_like(P)
    for i in range(ncomp):
        invPP[i] = nplinalg.inv(P[i])
    z1=np.dot(dRdth,Xt.T).T
    
    dpdth=np.zeros((npt,ncomp),dtype=dtype)
    dpdx=np.zeros((npt,ncomp),dtype=dtype)
    dpdy=np.zeros((npt,ncomp),dtype=dtype)
    for i in range(ncomp): 
        z2=Xn-MU[i]
        z3=np.dot(invPP[i],z2.T).T
        y1=np.multiply(z1,z3)
        dpdth[:,i] = -np.sum(y1,axis=1)
        dpdx[:,i] = -z3[:,0]
        dpdy[:,i] = -z3[:,1]
    
    a1 = -invp*np.sum(pcomp*dpdth,axis=1)
    a2 = -invp*np.sum(pcomp*dpdx,axis=1)
    a3 = -invp*np.sum(pcomp*dpdy,axis=1)
    dth=np.mean(a1)
    dx=np.mean(a2)
    dy=np.mean(a3)
    g=np.array([dth,dx,dy])
    return f,g





#%% Loop closure


@jit(float64(float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCost(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    y=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        y[s+1,:]= x[s*3:s*3+3]
    

    th=y[:,0] # global
    txy=y[:,1:] # global
    F=np.zeros(Hrels.shape[0]*3)

        
    for idx in numba.prange(Hrels.shape[0]):
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        fHi=getHmat(thi,ti)
        fHj=getHmat(thj,tj)
        
        jHf = nplinalg.inv(fHj)
        jHi_var=jHf.dot(fHi)
        
        tji_var,thji_var=extractPosAngle(jHi_var)
        
        if np.abs(i-j)<=5:
            c = 20
        else:
            c = 1
        
        F[idx*3] = c*(thji_var-thji)
        F[idx*3+1] = c*(tji_var[0]-tji[0])
        F[idx*3+2] = c*(tji_var[1]-tji[1])
        
    # f=np.sum(F)
    # f=1.3
    e = np.sum(F**2)
    return e

@jit(float64[:](float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCost_lsq(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    y=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        y[s+1,:]= x[s*3:s*3+3]
    

    th=y[:,0] # global
    txy=y[:,1:] # global
    F=np.zeros(Hrels.shape[0]*3)

        
    for idx in numba.prange(Hrels.shape[0]):
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        fHi=getHmat(thi,ti)
        fHj=getHmat(thj,tj)
        
        jHf = nplinalg.inv(fHj)
        jHi_var=jHf.dot(fHi)
        
        tji_var,thji_var=extractPosAngle(jHi_var)
        
        if np.abs(i-j)<=5:
            c = 20
        else:
            c = 1
        
        F[idx*3] = c*(thji_var-thji)
        F[idx*3+1] = c*(tji_var[0]-tji[0])
        F[idx*3+2] = c*(tji_var[1]-tji[1])
        
    # f=np.sum(F)
    # f=1.3
    return F

@jit(float64[:](float64[:],int32[:,:],float64[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCostHess_lsq(x,Hrelsidx,Hrels,Hessrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    y=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        y[s+1,:]= x[s*3:s*3+3]
    

    th=y[:,0] # global
    txy=y[:,1:] # global
    F=np.zeros(Hrels.shape[0]*3)

    hess = np.zeros((3,3))
    for idx in numba.prange(Hrels.shape[0]):
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        
        hess[0] = Hessrels[idx,0:3]
        hess[1] = Hessrels[idx,3:6]
        hess[2] = Hessrels[idx,6:9]
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        fHi=getHmat(thi,ti)
        fHj=getHmat(thj,tj)
        
        jHf = nplinalg.inv(fHj)
        jHi_var=jHf.dot(fHi)
        
        tji_var,thji_var=extractPosAngle(jHi_var)
        
        if np.abs(i-j)<=3:
            c = 30
        else:
            c = 1
        
        a = np.array([thji,tji[0],tji[1]])
        e=c*a.dot(hess.dot(a))
        
        F[idx*3:idx*3+3] = a*(hess.dot(a))
        # F[idx*3+1] = c*(tji_var[0]-tji[0])
        # F[idx*3+2] = c*(tji_var[1]-tji[1])
        
        
    # f=np.sum(F)
    # f=1.3
    return F
