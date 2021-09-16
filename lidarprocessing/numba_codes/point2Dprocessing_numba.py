# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc,boolean
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
from scipy.sparse import csr_matrix,lil_matrix

numba_cache=True
dtype=np.float64

#%% histogram bins









@jit(int32[:,:](float64[:,:], float64[:], float64[:]),nopython=True, nogil=True,cache=True) 
def numba_histogram2D(X, xedges,yedges):
    x_min = np.min(xedges)
    x_max = np.max(xedges)
    nx = len(xedges) 
    
    
    y_min = np.min(yedges)
    y_max = np.max(yedges)
    ny = len(yedges)
    
    H = np.zeros((nx-1,ny-1),dtype=np.int32)
    
    dxy=np.array([x_max-x_min,y_max-y_min])
    xymin = np.array([x_min,y_min])
    xymax = np.array([x_max,y_max])
    
    nxy = np.array([nx-1,ny-1])
    
       
    dd = nxy*((X-xymin)/dxy)

    for i in range(X.shape[0]):
        if np.all( (X[i]-xymin)>=0) and np.all( (xymax-X[i])>=0):
            H[int(dd[i][0]),int(dd[i][1])]+=1   

        
    return H


    
@jit(float32(float64[:], int32[:,:], float64[:,:], float64[:], float64[:], int32),nopython=True, nogil=True,cache=True) 
def binScanCost(x,P,Xt,xedges,yedges,cntThres):
    # x is the pose
    # P is the Probability of bins of the keyframe
    
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    
    H=numba_histogram2D(Xn, xedges,yedges)
    
    nx = len(xedges) 
    ny = len(yedges) 
    
    # Pn is the probability of scan points in the keyframe histogram
    # Pn=np.zeros((nx-1,ny-1),dtype=np.int32)
    Pn=np.sign(H)
    
    # for i in range(nx-1):
    #     for j in range(ny-1):
    #         if H[i,j]>=cntThres:
    #             Pn[i,j]=1

    mbin_and = np.sum(np.logical_and(P,Pn))
    # d2 = np.sum(np.logical_or(P,Pn))
    
    # dd = d.astype(np.int32)
    
    # f = np.sum(dd.reshape(-1))
    # mbin_and,mbin_or,Hist2_ovrlp=Commonbins(Hist1_ovrlp,Xn,xedges_ovrlp,yedges_ovrlp,1)
    activebins1_ovrlp = np.sum(P)
    activebins2_ovrlp = np.sum(Pn)
    
    
    # mbinfrac_ovrlp=mbin_and/mbin_or
    # mbinfrac_ActiveOvrlp = max([mbin_and/activebins1_ovrlp,mbin_and/activebins2_ovrlp])
    mbinfrac_ActiveOvrlp = mbin_and/activebins1_ovrlp

    return mbinfrac_ActiveOvrlp

@jit(numba.types.Tuple((int32[:,:],float64[:],float64[:]))(float64[:,:], float64[:,:], float64[:]),nopython=True, nogil=True,cache=True) 
def binScanEdges(Xb,Xt,dx):
    mn=np.zeros(2)
    mn[0] = np.min(Xb[:,0])
    mn[1] = np.min(Xb[:,1])
    
    mx=np.zeros(2)
    mx[0] = np.max(Xb[:,0])
    mx[1] = np.max(Xb[:,1])
    
    mnt=np.zeros(2)
    mnt[0] = np.min(Xt[:,0])
    mnt[1] = np.min(Xt[:,1])
    
    mxt=np.zeros(2)
    mxt[0] = np.max(Xt[:,0])
    mxt[1] = np.max(Xt[:,1])
    
    dt= np.max(0.5*(mxt-mnt))
    
    mn = mn-dt
    mx = mx+dt
    
    xedges = np.arange(mn[0],mx[0],dx[0])
    yedges = np.arange(mn[1],mx[1],dx[1])
    
    nx = len(xedges) 
    ny = len(yedges)
    H = numba_histogram2D(Xb, xedges,yedges)
    P=np.sign(H)
    
    return P, xedges,yedges

# @jit(numba.types.Tuple((int32[:,:],float64[:],float64[:]))(float64[:,:], float64[:,:], float64[:]),nopython=True, nogil=True,cache=True) 
# def posematchMetrics(X1,X2,H12,dx):
#     # transform points to X1 space, bin it and then compute the posematch cost
#     R=H12[0:2,0:2]
#     X12=H12.dot(X2.T)+H12[0:2,2]
    
            


@jit(numba.types.Tuple((float64[:],float64))(float64[:,:], int32[:,:], float64[:,:],float64[:],float64[:], int32),nopython=True, nogil=True,cache=True) 
def binScanMatcher(Posegrid,P,Xt,xedges,yedges,cntThres):
    # Xb are keyframe points
    # Xt are are scan points
    
    
    # P=np.zeros((nx-1,ny-1),dtype=np.int32)
    # for i in range(nx-1):
    #     for j in range(ny-1):
    #         if H[i,j]>=cntThres:
    #             P[i,j]=1
    
    
    
    m = np.zeros(Posegrid.shape[0])
    for i in numba.prange(Posegrid.shape[0]):
        m[i]=binScanCost(Posegrid[i],P,Xt,xedges,yedges,cntThres) 
        
    ind = np.argmax(m)  
        
    res = Posegrid[ind]
    
    return res,m[ind]

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

@jit(float64(float64, float64),nopython=True, nogil=True,parallel=False,cache=True) 
def anglediff(th1,th2):
    a = th1-th2
    if a>np.pi:
        a-=2*np.pi
    elif a<-np.pi:
        a+=2*np.pi
    elif a==-np.pi:
        a=np.pi
    
    return a


#%%
@jit(float64[:](float64[:,:], float64[:,:], float64[:,:,:],float64[:], float64[:,::1]),nopython=True, nogil=True,parallel=False,cache=True) 
def gridsearch_alignment(Posegrid,MU,P,W,X):
    
    m=np.zeros(Posegrid.shape[0])
    for i in numba.prange(Posegrid.shape[0]):
        m[i]=getcostalign(Posegrid[i],X,MU,P,W)
    ind = np.argmin(m)
    res = Posegrid[ind]
    return res



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
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    

    th=yy[:,0] # global
    txy=yy[:,1:] # global
    F=0

    S = np.identity(3)
    S[0,0]=1
    y=np.zeros(3)
    
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
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        
        y[0]=thji_var-thji
        y[1] = tji_var[0]-tji[0]
        y[2] = tji_var[1]-tji[1]
        
        
        F= F+0.5*y.dot(S.dot(y)) 

    return F

@jit(numba.types.Tuple((float64,float64[:]))(float64[:],int32[:,:],float64[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCost_Fjac(x,Hrelsidx,Hrels,Hessrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    

    th=yy[:,0] # global
    txy=yy[:,1:] # global
    
    m=Hrels.shape[0]*3
    n = len(x)
    
    dydxi = np.zeros((3,3))
    dydxj = np.zeros((3,3))
    
    F=0
    
    
    # S = np.identity(3)
    # S[0,0]=1
    
    for idx in numba.prange(Hrels.shape[0]):
        jac = np.zeros(len(x))
        y=np.zeros(3)
        hess = np.zeros((3,3))
                    
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        # ytrue = np.hstack([thji,tji])
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        hess[0] = Hessrels[idx,0:3]
        hess[1] = Hessrels[idx,3:6]
        hess[2] = Hessrels[idx,6:9]
        # S=np.identity(3)
        S = hess
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        

        # for i
        y[0]=thji_var-thji
        y[1] = tji_var[0]-tji[0]
        y[2] = tji_var[1]-tji[1]
        
        # y=np.hstack([thji_var-thji,tji_var-tji])
        
        dydxi[0,0]=1
        dydxi[0,1:]=0
        dydxi[1:,0]=0
        dydxi[1:,1:]=Rj.T
        
        # dJjidi = nplinalg.multi_dot([dydxi.T,S,y]) 
        dJjidi = dydxi.T.dot(S.dot(y)) 
        
        
        # for j
        dRjTdth = np.array([[-np.sin(thj),np.cos(thj)],[-np.cos(thj),-np.sin(thj)]])
        
        dydxj[0,0]=-1
        dydxj[0,1:]=0
        dydxj[1:,0]=dRjTdth.dot(ti-tj)
        dydxj[1:,1:]=-Rj.T
        
        dJjidj = dydxj.T.dot(S.dot(y))  
        #np.matmul(dydxj.T,y)
        if i>0:
            pp=i-1
            jac[3*pp:3*pp+3] = jac[3*pp:3*pp+3]+dJjidi
        
        if j>0:
            pp=j-1
            jac[3*pp:3*pp+3] = jac[3*pp:3*pp+3]+dJjidj
            
        # jac[3*j:3*j+3] = jac[3*j:3*j+3]+dJjidj

        

        F= F+0.5*y.dot(S.dot(y)) 
    
    return F,jac
        
    

@jit(float64[:](float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCost_lsq(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    

    th=yy[:,0] # global
    txy=yy[:,1:] # global
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
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        
        if np.abs(i-j)<=2:
            c=10
        else:
            c=1


        
        F[idx*3] = c*anglediff(thji_var,thji)
        F[idx*3+1] = c*(tji_var[0]-tji[0])
        F[idx*3+2] = c*(tji_var[1]-tji[1])
        
        
    # f=np.sum(F)
    # f=1.3
    return F

# @jit(float64[:,:](float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCost_lsq_jac(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    
    # now yy includes the first frame with 0,0,0 fixed pose
    th=yy[:,0] # global
    txy=yy[:,1:] # global
    
    m=Hrels.shape[0]*3
    n = len(x)
    jac=lil_matrix((m, n), dtype=dtype)  
    # jac=np.zeros((m, n), dtype=dtype)  
    
    dydxi = np.zeros((3,3))
    dydxj = np.zeros((3,3))
    y=np.zeros(3)
    # print("Hrelsidx")
    # print(Hrelsidx)
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
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        

        # for i
        y[0]=thji_var-thji
        y[1] = tji_var[0]-tji[0]
        y[2] = tji_var[1]-tji[1]
        
        # np.hstack([thji_var,tji_var])
        
        dydxi[0,0]=1
        dydxi[0,1:]=0
        dydxi[1:,0]=0
        dydxi[1:,1:]=Rj.T
        
        # dJjidi = np.matmul(dydxi.T,y)
        
        
        # for j
        dRjTdth = np.array([[-np.sin(thj),np.cos(thj)],[-np.cos(thj),-np.sin(thj)]])
        
        dydxj[0,0]=-1
        dydxj[0,1:]=0
        dydxj[1:,0]=dRjTdth.dot(ti-tj)
        dydxj[1:,1:]=-Rj.T
        
        if np.abs(i-j)<=2:
            c=10
        else:
            c=1
            
        # dJjidj = np.matmul(dydxj.T,y)
        if i>0:
            pp=i-1
            jac[3*idx,3*pp:3*pp+3] = c*dydxi[0] 
            jac[3*idx+1,3*pp:3*pp+3] =  c*dydxi[1]
            jac[3*idx+2,3*pp:3*pp+3] =  c*dydxi[2]
        
        # print("m,n,idx,i,j,Hrels.shape[0]",m,n,idx,i,j,Hrels.shape[0])
        if j>0:
            pp=j-1
            jac[3*idx,3*pp:3*pp+3] =  c*dydxj[0] 
            jac[3*idx+1,3*pp:3*pp+3] =  c*dydxj[1]
            jac[3*idx+2,3*pp:3*pp+3] =  c*dydxj[2]

        
        # F[idx*3] = (thji_var-thji)
        # F[idx*3+1] = (tji_var[0]-tji[0])
        # F[idx*3+2] = (tji_var[1]-tji[1])
        
        
        
    # f=np.sum(F)
    # f=1.3
    return jac

@jit(float64[:](float64[:],int32[:,:],float64[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCostHess_lsq(x,Hrelsidx,Hrels,Hessrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    

    th=yy[:,0] # global
    txy=yy[:,1:] # global
    F=np.zeros(Hrels.shape[0])

    hess = np.zeros((3,3))
    y=np.zeros(3)
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
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        

        

        
        e = np.array([thji_var-thji,tji_var[0]-tji[0],tji_var[1]-tji[1]])
        e=np.sqrt(np.abs(e))
        y=hess.dot(e)
        y=e*y
        # F[idx] = y
        F[idx*3] = y[0]
        F[idx*3+1] = y[1]
        F[idx*3+2] = y[2]

        
        
    # f=np.sum(F)
    # f=1.3
    return F

@jit(float64[:,:](float64[:],int32[:,:],float64[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCostHess_lsq_jac(x,Hrelsidx,Hrels,Hessrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    
    # now yy includes the first frame with 0,0,0 fixed pose
    th=yy[:,0] # global
    txy=yy[:,1:] # global
    
    m=Hrels.shape[0]*3
    n = len(x)
    # jac=lil_matrix((m, n), dtype=dtype)  
    jac=np.zeros((m, n), dtype=dtype)  
    
    dydxi = np.zeros((3,3))
    dydxj = np.zeros((3,3))
    y=np.zeros(3)
    hess = np.zeros((3,3))
    
    # print("Hrelsidx")
    # print(Hrelsidx)
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
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        

        # for i
        y[0]=thji_var-thji
        y[1] = tji_var[0]-tji[0]
        y[2] = tji_var[1]-tji[1]
        
        # np.hstack([thji_var,tji_var])
        
        dydxi[0,0]=1
        dydxi[0,1:]=0
        dydxi[1:,0]=0
        dydxi[1:,1:]=Rj.T
        
        # dJjidi = np.matmul(dydxi.T,y)
        
        
        # for j
        dRjTdth = np.array([[-np.sin(thj),np.cos(thj)],[-np.cos(thj),-np.sin(thj)]])
        
        dydxj[0,0]=-1
        dydxj[0,1:]=0
        dydxj[1:,0]=dRjTdth.dot(ti-tj)
        dydxj[1:,1:]=-Rj.T
        
        # dJjidj = np.matmul(dydxj.T,y)
        if i>0:
            pp=i-1
            jac[3*idx,3*pp:3*pp+3] = dydxi[0] 
            jac[3*idx+1,3*pp:3*pp+3] =  dydxi[1]
            jac[3*idx+2,3*pp:3*pp+3] =  dydxi[2]
        
        # print("m,n,idx,i,j,Hrels.shape[0]",m,n,idx,i,j,Hrels.shape[0])
        if j>0:
            pp=j-1
            jac[3*idx,3*pp:3*pp+3] =  dydxj[0] 
            jac[3*idx+1,3*pp:3*pp+3] =  dydxj[1]
            jac[3*idx+2,3*pp:3*pp+3] =  dydxj[2]

        
        # F[idx*3] = (thji_var-thji)
        # F[idx*3+1] = (tji_var[0]-tji[0])
        # F[idx*3+2] = (tji_var[1]-tji[1])
        
        
        
    # f=np.sum(F)
    # f=1.3
    return jac
