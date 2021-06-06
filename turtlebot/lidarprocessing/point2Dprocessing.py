# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:11:58 2020

@author: nadur
"""

# from numba import set_num_threads,config, njit, threading_layer,get_num_threads

# set_num_threads(1)
# set the threading layer before any parallel target compilation
# config.THREADING_LAYER = 'threadsafe'
# config.NUMBA_NUM_THREADS

# print("#Threads: %d" % get_num_threads())
# print("Threading layer chosen: %s" % threading_layer())

import numpy as np
import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,float32
from numba import njit, prange,jit

import multiprocessing as mp
import threading
import queue

import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc

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
numba_cache=False


dtype=np.float64
#%% Submap- grid



    
def binnerDownSampler(X,dx=0.05,cntThres=1):
    mn = np.min(X,axis=0)
    mx = np.max(X,axis=0)
    mn = mn-dx/2
    mx = mx+dx/2
    
    xedges = np.arange(mn[0],mx[0]+dx,dx)
    yedges = np.arange(mn[1],mx[1]+dx,dx)
    
    H, xedges, yedges = np.histogram2d(X[:,0],X[:,1],bins=(xedges, yedges) )
    
    inds = np.argwhere(H>=cntThres)   
    x=xedges[inds[:,0]]+dx/2
    y=yedges[inds[:,1]]+dx/2                                
    Xd = np.vstack([x,y]).T
    
    return Xd






    
#%% Pose estimation by keyframe
# - manage keyframes
# - estimate pose to closest keyframe

def get0meanIcov(X):
    m=np.mean(X,axis=0)
    Xd=X-m
    return Xd,m

def wcost(W,pcomp,randind):
    pp=np.sum(pcomp*W,axis=1)
    c=0
    for i,j in randind:
        c = c + (pp[i]-pp[j])**4
        # c = c + (np.log(pp[i])-np.log(pp[i-1]))**2
    return c



def getclf(X,params,doReWtopt=True,means_init=None,weights_init=None,precisions_init=None):
    
    # Xdb=binnerDownSampler(X,dx=0.025,cntThres=1)
    Xdb = X
    
    clf = mixture.GaussianMixture(n_components=params['n_components'],
                                  means_init=means_init, 
                                  weights_init=weights_init,
                                  precisions_init=precisions_init,
                                  covariance_type='full',reg_covar=params['reg_covar'])
    clf.fit(Xdb)

    
    MU=np.ascontiguousarray(clf.means_,dtype=dtype)
    P=np.ascontiguousarray(clf.covariances_,dtype=dtype)
    W=np.ascontiguousarray(clf.weights_,dtype=dtype)
    
    ncomp = MU.shape[0]
    dim = MU.shape[1]
    npt = X.shape[0]
    invPP=np.zeros_like(P)
    for i in range(ncomp):
        invPP[i] = nplinalg.inv(P[i])
    
    if doReWtopt:
        pcomp=uqgmmfnc.gmm_evalcomp_fast(X,MU,P,W)
        bnds=[]
        for i in range(len(W)):
            bnds.append((0,1000))
        randind = []
        for i in range(100):
            a=np.random.choice(X.shape[0], 2)
            randind.append(a)
            
        res = minimize(wcost, W,args=(pcomp,randind),jac=False,bounds=bnds ,method='SLSQP', tol=1e-4) # 'Nelder-Mead'
        # res = least_squares(wcost_lsq, W,args=(pcomp,),bounds=([0]*len(W),[1000]*len(W)) )
        res.x=res.x/np.sum(res.x)
          
        W=res.x
        clf.weights_=res.x
    

    res={'clf':clf,'success':True}
    return res

#%%


def getgridvec(thset,txset,tyset):
    th,x,y=np.meshgrid(thset,txset,tyset)
    Posegrid = np.vstack([th.ravel(), x.ravel(), y.ravel()]).T
    Posegrid=np.ascontiguousarray(Posegrid,dtype=dtype)
    return Posegrid


#%%



def gridsearch_alignment_opt(Posegrid,MU,P,W,X):
    

    fres=None
    M=1e10
    for i in range(Posegrid.shape[0]):
       res = minimize(nbpt2Dproc.getcostgradient, Posegrid[i],args=(X,MU,P,W),jac=True ,method='BFGS', tol=1e-4) # 'Nelder-Mead'            
       if res.fun < M and res.success:
           M=res.fun
           fres= res

    th=fres.x[0]
    t=fres.x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=dtype,order='C')

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=fres.fun
    
    # hess_inv is like covariance
    return sHk,err, fres.hess_inv

# @jit(numba.types.Tuple((float64[:,:],float64[:],float64[:,:]))(float64[:], float64[:,::1], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=True) 
def gridsearch_alignment_brute(Posegrid,MU,P,W,X):
    

    fres=None
    M=1e10

    fres=nbpt2Dproc.gridsearch_alignment(Posegrid,MU,P,W,X)

    fres = minimize(nbpt2Dproc.getcostgradient, fres,args=(X,MU,P,W),jac=True ,method='BFGS', tol=1e-4)
  
    th=fres.x[0]
    t=fres.x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=dtype,order='C')

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=fres.fun
    
    # hess_inv is like covariance
    return sHk,err, fres.hess_inv


#%%
def alignscan2keyframe(MU,P,W,X):
    best=np.zeros(3)
    # Posegrid = pt2dproc.getgridvec(thset,txset,tyset)
    # best=gridsearch_alignment(Posegrid,MU,P,W,X)
    # print("best = ",best) 
    
    res = minimize(nbpt2Dproc.getcostgradient, best,args=(X,MU,P,W),jac=True ,method='BFGS', tol=1e-4) # 'Nelder-Mead'
    
    
    th=res.x[0]
    t=res.x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=dtype,order='C')

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=res.fun
    
    # hess_inv is like covariance
    return sHk,err, res.hess_inv

def scan2keyframe_bin_match(Xclf,X,Posegrid,dx=1,sHk=np.identity(3)):
    kHs = nplinalg.inv(sHk)
    Xk=np.matmul(kHs,np.vstack([X.T,np.ones(X.shape[0])])).T  
    Xd=Xk[:,:2]
    # Xd=Xk-m_clf
    # MU=np.ascontiguousarray(KeyFrameClf.means_,dtype=dtype)
    # P=np.ascontiguousarray(KeyFrameClf.covariances_,dtype=dtype)
    # W=np.ascontiguousarray(KeyFrameClf.weights_,dtype=dtype)
    Xd = np.ascontiguousarray(Xd,dtype=dtype) 
    

    res,mbin = nbpt2Dproc.binScanMatcher(Posegrid,Xclf,Xd,dx,1)
    
    
    # # th=res[0]
    # # t=np.array([res[1],res[2]])
    # # err=0
    # # hess_inv=np.identity(3)
    
    # fres = minimize(nbpt2Dproc.getcostgradient, res,args=(Xd,MU,P,W),jac=True ,method='BFGS', tol=1e-4)
    
    th=res[0]
    t=res[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=dtype,order='C')

    kHs_rel=np.hstack([R,t.reshape(2,1)])
    kHs_rel = np.vstack([kHs_rel,[0,0,1]])
    sHk_rel = nplinalg.inv(kHs_rel)
    # err=fres.fun
    # hess_inv = fres.hess_inv
    

    sHk_corrected=np.matmul(sHk,sHk_rel)
    
    return sHk_corrected,mbin

    
def scan2keyframe_match(KeyFrameClf,Xclf,X,params,sHk=np.identity(3)):
    # sHrelk is the transformation from k to s
    
        
    kHs = nplinalg.inv(sHk)
    Xk=np.matmul(kHs,np.vstack([X.T,np.ones(X.shape[0])])).T  
    Xd=Xk[:,:2]
    # Xd=Xk-m_clf
    MU=np.ascontiguousarray(KeyFrameClf.means_,dtype=dtype)
    P=np.ascontiguousarray(KeyFrameClf.covariances_,dtype=dtype)
    W=np.ascontiguousarray(KeyFrameClf.weights_,dtype=dtype)
    Xd = np.ascontiguousarray(Xd,dtype=dtype) 
    
    sHk_rel,err,hess_inv = alignscan2keyframe(MU,P,W,Xd)
    
    xy_hess_inv_eigval,xy_hess_inv_eigvec = nplinalg.eig( hess_inv[1:,1:] )
    th_hess_inv = hess_inv[0,0]
    txy,theta = nbpt2Dproc.extractPosAngle(sHk_rel)
    
    
    # if np.max(xy_hess_inv_eigval) > params['xy_hess_inv_thres']:
    #     print("going to bin search refinement")
    #     ind = np.argmax(xy_hess_inv_eigval)
    #     vec = xy_hess_inv_eigvec[:,ind]
        
    #     PP = np.zeros_like(P)
    #     for i in range(P.shape[0]):
    #         # eigval,eigvec = nplinalg.eig( P[i] )
    #         # if eigvec[:,0].dot(vec)> eigvec[:,1].dot(vec):
    #         #     eigvec[:,0]=eigvec[:,0]*2**2
    #         # else:
    #         #     eigvec[:,1]=eigvec[:,1]*2**2
    #         # PP[i] = eigvec.dot(np.diag(eigval).dot(eigvec.T))
    #         PP[i] = P[i]*2**2
    #     # sHk_rel,err,hess_inv = alignscan2keyframe(MU,P,W,Xd)
        
    #     pose=np.identity(3)
    #     E=err
    #     for i in np.linspace(-5,5,11):
    #         # tt=txy+vec*i
    #         # Posegrid.append([theta,tt[0],tt[1]]) 
    #         SS = sHk_rel.copy()
    #         # print(SS.shape)
    #         SS[0:2,2] = SS[0:2,2]+vec*i
    #         # print(SS.shape)
    #         # print(Xd.shape)
    #         XX=np.matmul(SS,np.vstack([Xd.T,np.ones(Xd.shape[0])])).T  
    #         XX=XX[:,:2]
    #         XX = np.ascontiguousarray(XX,dtype=dtype) 
    #         sHk_inter,err_inter,hess_inv_inter = alignscan2keyframe(MU,PP,W,XX)
    #         if err_inter<E:
    #             pose = sHk_inter
    #             E=err_inter
    #         fig=plt.figure()
    #         ax = fig.add_subplot(111)
    #         ax.plot(Xclf[:,0],Xclf[:,1],'r.')
    #         ax.plot(XX[:,0],XX[:,1],'bo')
    #         eigval,eigvec = nplinalg.eig( hess_inv_inter[1:,1:] )
    #         ax.set_title("err=%f , err_inter= %f, eigval=%f,%f"%(err,err_inter,eigval[0],eigval[1]))
    #         plt.show()
            
    #     sHk_rel =np.matmul(sHk_rel,pose) 
        
        # Posegrid = np.array(Posegrid)
        
        # res,m = nbpt2Dproc.binScanMatcher(Posegrid,Xclf,Xd,0.2,1)
        
        
        # th=fres.x[0]
        # t=fres.x[1:]
        # R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=dtype,order='C')
        
        # # pdb.set_trace()    
        
        # kHs_rel=np.hstack([R,t.reshape(2,1)])
        # kHs_rel = np.vstack([kHs_rel,[0,0,1]])
        # sHk_rel = nplinalg.inv(kHs_rel)
        # err=fres.fun
        # hess_inv = fres.hess_inv
        
   
            
    sHk_corrected=np.matmul(sHk,sHk_rel)
    
    return sHk_corrected,err,hess_inv


def Commonbins(P,Xn,xedges,yedges,cntThres):
    H=nbpt2Dproc.numba_histogram2D(Xn, xedges,yedges)
    
    nx = len(xedges) 
    ny = len(yedges) 
    
    
    # Pn is the probability of scan points in the keyframe histogram
    Pn=np.zeros((nx-1,ny-1),dtype=np.int32)
    
    
    Pn[H>=cntThres]=1

    dand = np.logical_and(P,Pn)
    dor = np.logical_or(P,Pn)
    # dd = d.astype(np.int32)
    
    Fand = np.sum(dand.reshape(-1))
    For = np.sum(dor.reshape(-1))
    return Fand,For,Pn


def poseGraph_keyFrame_matcher(poseGraph,poseData,idx1,idx2,params):
    # fromidx is idx1, toidx is idx2 
    clf1=poseGraph.nodes[idx1]['clf']
    # m_clf1=poseGraph.nodes[idx1]['m_clf']
    sHg_1 = poseGraph.nodes[idx1]['sHg']
    sHg_2 = poseGraph.nodes[idx2]['sHg']
    
    H21_est = np.matmul(sHg_2,nplinalg.inv(sHg_1))
    
    # H21_est = np.identity(3)
    T1 = poseGraph.nodes[idx1]['time']
    T2 = poseGraph.nodes[idx2]['time']
    X1 = poseData[T1]['X']
    X2 = poseData[T2]['X']
    
    # do a gmm fit first and check overlapp err
    H21,err,hess_inv=scan2keyframe_match(clf1,X1,X2,params,sHk=H21_est)
    
    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    
    xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.getEdges(X1,dxcomp)
    Hist1_ovrlp = nbpt2Dproc.numba_histogram2D(X1, xedges_ovrlp,yedges_ovrlp)
    Hist1_ovrlp[Hist1_ovrlp>=1]=1
    Hist1_ovrlp[Hist1_ovrlp<1]=0
    activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    
    H12 = nplinalg.inv(H21)
    Xn=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
    Xn=Xn[:,:2]    
    mbin_and,mbin_or,Hist2_ovrlp=Commonbins(Hist1_ovrlp,Xn,xedges_ovrlp,yedges_ovrlp,1)
    activebins2_ovrlp = np.sum(Hist2_ovrlp.reshape(-1))
    # mbinfrac_ovrlp=max([mbin_ovrlp/activebins1_ovrlp,mbin_ovrlp/activebins2_ovrlp])
    mbinfrac_ovrlp=mbin_and/mbin_or
    
    if mbinfrac_ovrlp>=params['LOOPCLOSE_BIN_MIN_FRAC']:
        return H21,err,mbin_and,mbinfrac_ovrlp,hess_inv
    
    
    mx=np.max(X1,axis=0)
    mn=np.min(X1,axis=0)
    r = np.max(nplinalg.norm(X2,axis=1))

    dx=params['LOOPCLOSE_BIN_MATCHER_dx']
    L=params['LOOPCLOSE_BIN_MATCHER_L']

    dth = 1*dx/r
    dxx = 0.7*dx 
 
    
    # thset = np.arange(-np.pi/3,np.pi/3+dth,dth)
    # txset = np.arange(-L,L+dxx,dxx) # 
    # tyset = np.arange(-L,L+dxx,dxx) #
    
    thset = np.arange(-np.pi/6,np.pi/6+dth,dth)
    txset = np.arange(-L,L+dxx,dxx) # 
    tyset = np.arange(-L,L+dxx,dxx) #
    
    Posegrid = getgridvec(thset,txset,tyset)
    # print("Posegrid = ",Posegrid.shape)
    
    H21,mbin = scan2keyframe_bin_match(X1,X2,Posegrid,dx=dx,sHk=H21)
    # H21[0:2,2] = H21[0:2,2]+dd
    
    if params['Do_BIN_DEBUG_PLOT-dx']:
        xedges,yedges=nbpt2Dproc.getEdges(X1,dx)
        Hist1 = nbpt2Dproc.numba_histogram2D(X1, xedges,yedges)
        Hist1[Hist1>=1]=1
        Hist1[Hist1<1]=0
        print("first Hist1=",np.sum(Hist1.reshape(-1)))
        print("first mbin=",mbin)
        H12 = nplinalg.inv(H21)
        X12=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
        X12=X12[:,:2]
        
        fig=plt.figure('Do_BIN_DEBUG_PLOT-dx')
        ax=fig.add_subplot(111)
        ax.pcolor(xedges,yedges,Hist1.T,shading='flat',alpha=0.4 )
        ax.plot(X1[:,0],X1[:,1],'r.')
        ax.plot(X12[:,0],X12[:,1],'bo')
        ax.axis('equal')
        
    
    
    if params['Do_BIN_FINE_FIT']:
        # txy,theta = nbpt2Dproc.extractPosAngle(H21)
        
        # X12=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
        # X12=X12[:,:2]
        
        thset = np.linspace(-2*dx/r,2*dx/r,11)
        txset = np.arange(-dx,dx+0.14,0.14) # 
        tyset = np.arange(-dx,dx+0.14,0.14) #
        Posegrid = getgridvec(thset,txset,tyset)
        H21,mbin = scan2keyframe_bin_match(X1,X2,Posegrid,dx=0.16,sHk=H21)
    
    
    
    
    
    if params['Do_GMM_FINE_FIT']:
        H21,err,hess_inv=scan2keyframe_match(clf1,X1,X2,params,sHk=H21)
    
        
        
        
    H12 = nplinalg.inv(H21)
    Xn=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
    Xn=Xn[:,:2]    
    mbin_and,mbin_or,Hist2_ovrlp=Commonbins(Hist1_ovrlp,Xn,xedges_ovrlp,yedges_ovrlp,1)
    activebins2_ovrlp = np.sum(Hist2_ovrlp.reshape(-1))
    # mbinfrac_ovrlp=max([mbin_ovrlp/activebins1_ovrlp,mbin_ovrlp/activebins2_ovrlp])
    mbinfrac_ovrlp=mbin_and/mbin_or
    
    if params['Do_GMM_FINE_FIT'] is False:
        err=1-mbinfrac_ovrlp
        hess_inv = 0.5*np.identity(3)
        
        
    if params['Do_BIN_DEBUG_PLOT']:
        H12 = nplinalg.inv(H21)
        X12=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
        X12=X12[:,:2]
        
        fig=plt.figure('Do_BIN_DEBUG_PLOT')
        ax=fig.add_subplot(111)
        ax.pcolor(xedges_ovrlp,yedges_ovrlp,Hist1_ovrlp.T,shading='flat',alpha=0.4 )
        ax.plot(X1[:,0],X1[:,1],'r.')
        ax.plot(X12[:,0],X12[:,1],'bo')
        ax.axis('equal')

    
    
    

    # mbinfrac=mbin/activebins1
    
    return H21,err,mbin_and,mbinfrac_ovrlp,hess_inv





#%% poseGraph manipulation

def addedge2midscan(poseGraph,poseData,idx,KeyFrame_prevIdx,sHk,params,keepOtherScans=False):
    """
    idx : current keyframe id
    KeyFrame_prevIdx: previous keyframe id
    
    Find the mid frame (which should be a scan frame)
    
    """
    Lidxs = list(poseGraph.successors(KeyFrame_prevIdx))
    Lidxs = list(filter(lambda x: poseGraph.nodes[x]['frametype']!="keyframe",Lidxs)) # get all non-keyframes
    if len(Lidxs)>0:
        scanid = Lidxs[int(np.floor(len(Lidxs)/2))]
        # estimate pose to new keyframe
        Tscanid = poseGraph.nodes[scanid]['time']
        Xs = poseData[Tscanid]['X']

        KeyFrameClf = poseGraph.nodes[idx]['clf']
        Tidx = poseGraph.nodes[idx]['time']
        Xclf = poseData[Tidx]['X']
        msHk = poseGraph.edges[KeyFrame_prevIdx,scanid]['H']
        sHk_newkeyframe =  np.matmul(msHk,nplinalg.inv(sHk))
        
        sHk_corrected,err,hess_inv = scan2keyframe_match(KeyFrameClf,Xclf,Xs,params,sHk=sHk_newkeyframe)
        
        # if np.isfinite(err)==0:
        #     plotcomparisons_posegraph(poseGraph,idx,scanid,H12=nplinalg.inv( sHk_newkeyframe) )
        #     plotcomparisons(scanid,idx,H12=sHk_newkeyframe)
            
        poseGraph.add_edge(idx,scanid,H=sHk_corrected,H_prevframe=sHk_newkeyframe,err=err,hess_inv=hess_inv,edgetype="Key2Scan",color='r')
        
        # delete rest of the scan-ids as they are useless
        if keepOtherScans is False:
            for i in Lidxs:
                if i!=scanid and poseGraph.in_degree(i)==1:
                    poseGraph.remove_node(i)
                    

#%% features


def get2DptFeat(X,bins=np.arange(0,11,0.25)):
    db=bins[1]-bins[0]
    D=fastdist.matrix_pairwise_distance(X, fastdist.euclidean, "euclidean", return_matrix=False)
    D = D.reshape(1,-1)
    h,b=np.histogram(D, bins=bins)
    h=h/np.sum(db*h)
    return h
    


#%% bundle adjustment
def loopExists(poseGraph,idx1,idx2):
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    Lkey.sort()
    
    for idx in Lkey:
        if poseGraph.in_degree(idx)>1:
            return True

        
# def detectLoopClosures(poseGraph,idx,LOOP_CLOSURE_D_THES,LOOP_CLOSURE_POS_THES,LOOP_CLOSURE_ERR_THES,returnCopy=False):
#     """
#     idx is index of current "keyframe"
#     idx is the current pose. detect loop closures to all previous key frames

#     """
#     Lkeyprevious = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
#     Lkeyprevious = list(filter(lambda i: i<idx, Lkeyprevious))
    
#     h1=poseGraph.nodes[idx]['h']
#     p1=poseGraph.nodes[idx]['pos']
    
#     flag=False
#     for previdx in Lkeyprevious:
#         if previdx < idx-2 and poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:        
#             h2=poseGraph.nodes[previdx]['h']        
#             p2=poseGraph.nodes[previdx]['pos']
#             d=nplinalg.norm(h1-h2,ord=1)
#             if d<=LOOP_CLOSURE_D_THES and nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=LOOP_CLOSURE_POS_THES:
#                 # add loop closure edge
#                 print("Potential Loop closure")
#                 piHi,pi_err_i=poseGraph_keyFrame_matcher(poseGraph,idx,previdx)
#                 # iHpi,i_err_pi=poseGraph_keyFrame_matcher(poseGraph,previdx,idx)
#                 if pi_err_i < LOOP_CLOSURE_ERR_THES:
#                     poseGraph.add_edge(idx,previdx,H=piHi,edgetype="Key2Key-LoopClosure",d=d,color='b')
#                     print("Adding Loop closure")
#                 else:
#                     print("No loop closure this time")
    
#     if returnCopy:
#         return copy.deepcopy(poseGraph)
    
#     return poseGraph

def matchdetect(qin,qout,ExitFlag,Lkey,poseGraph,poseData,params) :
    while(True):
        idx = None
        try:
            idx = qin.get(True,0.2)
        except queue.Empty:
            idx = None
            
        if idx is not None:
            if poseGraph.nodes[idx]['LoopDetectDone'] is False:          
                h1=poseGraph.nodes[idx]['h']
                p1=poseGraph.nodes[idx]['pos']
                for previdx in Lkey:
                    if previdx < idx-2 and poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:
                        
                        h2=poseGraph.nodes[previdx]['h']
                        
                        p2=poseGraph.nodes[previdx]['pos']
                        d=nplinalg.norm(h1-h2,ord=1)
                        
                        c1 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=params['LOOP_CLOSURE_POS_THES']
                        c2 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)>=params['LOOP_CLOSURE_POS_MIN_THES']
                        if d<=params['LOOP_CLOSURE_D_THES'] and c1 and c2:
                            # add loop closure edge
                            # print("Potential Loop closure")
                            
                            piHi,pi_err_i,mbin,mbinfrac,hess_inv_err_i=poseGraph_keyFrame_matcher(poseGraph,poseData,idx,previdx,params)
                            # piHi,pi_err_i,hess_inv_err_i=0,0,0
                            a1=pi_err_i < params['LOOP_CLOSURE_ERR_THES']
                            a2=mbinfrac>=params['LOOPCLOSE_BIN_MIN_FRAC']
                            if a2:
                                qout.put(['edge',idx,previdx,piHi,pi_err_i,hess_inv_err_i,d])
                
                
                qout.put(['status',idx,'LoopDetectDone',True])
                
            
        if (ExitFlag.is_set() and qin.empty()):
            break
    
    print("thread done")              
        
def detectAllLoopClosures(poseGraph,poseData,params,returnCopy=False):
    """
    idx is index of current "keyframe"
    idx is the current pose. detect loop closures to all previous key frames

    """
    ctx = mp

    qin = ctx.Queue()
    qout = ctx.Queue()
    ExitFlag = ctx.Event()
    ExitFlag.clear()

    Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    
    Ncore = params['#ThreadsLoopClose']
    processes = []
    for i in range(Ncore):
        p = ctx.Process(target=matchdetect, args=(qin,qout,ExitFlag,Lkeys,poseGraph,poseData,params))
        processes.append( p )
        p.start()
        print("created thread")    
        time.sleep(0.01)  
    
    cnt=0
    for idx in Lkeys:
        # print(cnt,len(Lkeys))
        cnt+=1
        qin.put(idx)
        
    flg=0
    while True:
        res=None
        try:
            res=qout.get(block=True, timeout=0.1)
        except queue.Empty:
            time.sleep(0.01)
            
            
        if res is not None:
            if 'status' == res[0]:
                poseGraph.nodes[res[1]][res[2]] = res[3]
                print(res)
            elif 'edge' == res[0]:
                poseGraph.add_edge(res[1],res[2],H=res[3],err = res[4],hess_inv = res[5],edgetype="Key2Key-LoopClosure",d=res[6],color='b')
            
            flg=0
            
        
        
        if qin.empty() and qout.empty():
            ExitFlag.set()
            Palive=0
            for i in range(len(processes)):
                if processes[i].is_alive():
                    Palive+=1
            
            if Palive==0:
                break
    
    for i in range(len(processes)):
        print("Joining %d"%i)
        processes[i].join()
                
                
    if returnCopy:
        return copy.deepcopy(poseGraph)
    
    print("detect loop closes done")
    return poseGraph

def detectAllLoopClosures_noParallel(poseGraph,poseData,params,returnCopy=False):
    """
    idx is index of current "keyframe"
    idx is the current pose. detect loop closures to all previous key frames

    """


    Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    

    for idx in Lkeys:
        if poseGraph.nodes[idx]['LoopDetectDone'] is False:          
                h1=poseGraph.nodes[idx]['h']
                p1=poseGraph.nodes[idx]['pos']
                for previdx in Lkeys:
                    if previdx < idx-2 and poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:
                        
                        h2=poseGraph.nodes[previdx]['h']
                        
                        p2=poseGraph.nodes[previdx]['pos']
                        d=nplinalg.norm(h1-h2,ord=1)
                        
                        c1 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=params['LOOP_CLOSURE_POS_THES']
                        c2 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)>=params['LOOP_CLOSURE_POS_MIN_THES']
                        if d<=params['LOOP_CLOSURE_D_THES'] and c1 and c2:
                            # add loop closure edge
                            # print("Potential Loop closure")
                            
                            piHi,pi_err_i,mbin,mbinfrac,hess_inv_err_i=poseGraph_keyFrame_matcher(poseGraph,poseData,idx,previdx,params)
                            # piHi,pi_err_i,hess_inv_err_i=0,0,0
                            
                            a1=pi_err_i < params['LOOP_CLOSURE_ERR_THES']
                            a2=mbinfrac>=params['LOOPCLOSE_BIN_MIN_FRAC']
                            if a2:
                                poseGraph.add_edge(idx,previdx,H=piHi,err = pi_err_i,hess_inv = hess_inv_err_i,edgetype="Key2Key-LoopClosure",d=d,color='b')
                                
                                
                
                poseGraph.nodes[idx]['LoopDetectDone'] = True

        
    if returnCopy:
        return copy.deepcopy(poseGraph)
    
    print("detect loop closes done")
    return poseGraph


def adjustPoses(poseGraph,idx1,idx2,maxiter=10000,algo='BFGS'):
    # adjust the bundle from idx1 to idx2
    # i.e. optimize the global pose matrices, with constraint local poses 
    # idx1 is assumed to be fixed (like origin)
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    Lkey.sort()
    
    Lkey_dict={x:i for i,x in enumerate(Lkey)}
    
    Hrels=[] # Hrels=[[i,j,thji,txji,tyji],...]
    Hrelsidx=[]
    Hessrels=[] #Hessrels=[[hess_flattened],...]

    for j in range(1,len(Lkey)):
        idx = Lkey[j]
        for idx_prev in poseGraph.predecessors(idx):
            if idx_prev not in Lkey_dict:
                continue
            i = Lkey_dict[idx_prev]
            jHi=poseGraph.edges[idx_prev,idx]['H']
            j_hess_inv_i = poseGraph.edges[idx_prev,idx]['hess_inv']
            j_hess_i = nplinalg.inv(j_hess_inv_i)
            tji,thji = nbpt2Dproc.extractPosAngle(jHi)
            Hrels.append([thji,tji[0],tji[1] ])        
            Hessrels.append(j_hess_i.reshape(-1))
            Hrelsidx.append([i,j])
            
    Hrels = np.array(Hrels,dtype=dtype)
    Hessrels = np.array(Hessrels,dtype=dtype)

    Hrelsidx = np.array(Hrelsidx,dtype=np.int32)

    # pdb.set_trace()
    
    # x0=x0.reshape(-1)
    x0 = np.zeros((len(Lkey)-1,3))   #as first frame is 0,0,0          
    firstHg=poseGraph.nodes[Lkey[0]]['sHg']
    gHfirst = nplinalg.inv(firstHg)
    sHg_original={}
    bounds=[[-np.pi/4,np.pi/4],[-2,2],[-2,2]]*(len(Lkey)-1)
    for j in range(1,len(Lkey)):
        idx = Lkey[j]
        sHg=poseGraph.nodes[idx]['sHg']
        sHg_original[idx] = sHg
        
        jHf=np.matmul(sHg,gHfirst) # first to j (local global frame)
        fHj=nplinalg.inv(jHf)
        tj,thj = nbpt2Dproc.extractPosAngle(fHj)
        # tj = Hjf[:2,2]  
        x0[j-1,0]=thj
        x0[j-1,1:]=tj
        
        bounds[(j-1)*3][0]=thj+bounds[(j-1)*3][0]
        bounds[(j-1)*3][1]=thj+bounds[(j-1)*3][1]
        
        bounds[(j-1)*3+1][0]=tj[0]+bounds[(j-1)*3+1][0]
        bounds[(j-1)*3+1][1]=tj[0]+bounds[(j-1)*3+1][1]
        
        bounds[(j-1)*3+2][0]=tj[1]+bounds[(j-1)*3+2][0]
        bounds[(j-1)*3+2][1]=tj[1]+bounds[(j-1)*3+2][1]
        
    x0=x0.reshape(-1)
    # x0 = x0+0.01*np.random.randn(len(x0))
    
    lb = np.array([bnd[0] for bnd in bounds])
    ub = np.array([bnd[1] for bnd in bounds])
    
    # print("len(x) = ",len(x0))
    # print("Hrelsidx = ",Hrelsidx.shape)
    # print("Hrels = ",Hrels.shape)
    
    # pdb.set_trace()
    # print("starting cost globalPoseCost= ",nbpt2Dproc.globalPoseCost(x0,Hrelsidx,Hrels))
    # gg = nbpt2Dproc.globalPoseCost_lsq(x0,Hrelsidx,Hrels)
    # print("starting cost globalPoseCost_lsq= ",np.sum(gg**2))
    # st=time.time()
    # res = minimize(globalPoseCost, x0,args=(Hrelsidx,Hrels) ,method='BFGS', tol=1e-5)
    # success=res.success
    # et=time.time()
    # print(res.fun," time: ",et-st)
    # print(idx1,idx2,Hrelsidx,Hrels)
    # print(Hrelsidx.shape)
    st=time.time()
    if algo in ['lm','trf']:
        # res = least_squares(nbpt2Dproc.globalPoseCostHess_lsq, x0,args=(Hrelsidx,Hrels,Hessrels),method=algo)
        res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels),method=algo)
        
        # res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels),jac=nbpt2Dproc.globalPoseCost_lsq_jac ,method='trf')
    # res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels),jac=nbpt2Dproc.globalPoseCost_lsq_jac ,method='lm',xtol=1e-4,ftol=1e-4) #max_nfev
    # res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels),method='lm')
    
    # res = least_squares(nbpt2Dproc.globalPoseCostHess_lsq, x0,args=(Hrelsidx,Hrels,Hessrels) ,method='lm')
    # res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels) ,method='trf',loss='huber')
    # res = least_squares(nbpt2Dproc.globalPoseCostHess_lsq, x0,args=(Hrelsidx,Hrels,Hessrels) ,method='trf',loss='huber')
    if algo in ['BFGS']:
        res = minimize(nbpt2Dproc.globalPoseCost_Fjac, x0,args=(Hrelsidx,Hrels,Hessrels) ,jac=True,tol=1e-6,method=algo,options={'maxiter':maxiter}) #
    
    # res = minimize(nbpt2Dproc.globalPoseCost, x0,args=(Hrelsidx,Hrels) ,method='BFGS', tol=1e-4,options={'maxiter':3})
    
    
    
    success=res.success
    et=time.time()
    print(" time loop-closure-opt: ",et-st)


    
    x = res.x
    x=x.reshape(-1,3)   
    sHg_updated={}
    for j in range(1,len(Lkey)):     
        thj=x[j-1,0] # global
        tj=x[j-1,1:] # global
    
        fHj=nbpt2Dproc.getHmat(thj,tj)
        jHf=nplinalg.inv(fHj)
        sHg_updated[Lkey[j]] = np.matmul(jHf,firstHg)
        
    return res,sHg_updated,sHg_original

def updateGlobalPoses(poseGraph,sHg_updated):
    # loop closure optimization
    # Lkeycycle = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))    
    # Lkeycycle=list(nx.simple_cycles(poseGraph.subgraph(Lkeycycle)))
    # Lkeycycle=[sorted(m) for m in Lkeycycle]
    # Lkeycycle.sort(key=lambda x: len(x))
    


    for ns in sHg_updated.keys():
        sHg = sHg_updated[ns]
        gHs=nplinalg.inv(sHg)
        tpos=np.matmul(gHs,np.array([0,0,1]))
        poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
        poseGraph.nodes[ns]['sHg'] = sHg
        
    # now update other non-keyframes
    for ns in poseGraph.nodes:
        if poseGraph.nodes[ns]['frametype']!="keyframe":
            for pidx in poseGraph.predecessors(ns):
                if poseGraph.nodes[pidx]['frametype']=="keyframe": # and pidx in sHg_updated
                    psHg=poseGraph.nodes[pidx]['sHg']
                    nsHps=poseGraph.edges[pidx,ns]['H']
                    nsHg = nsHps.dot(psHg)
                    poseGraph.nodes[ns]['sHg']=nsHg
                    gHns=nplinalg.inv(nsHg)
                    tpos=np.matmul(gHns,np.array([0,0,1]))
                    poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
        
                    break
    
    return poseGraph


def adjustPoses_byPtAlign(posegraph,idx1,idx2):
    # adjust the bundle from idx1 to idx2
    # i.e. optimize the global pose matrices, with Gaussian overlap with scan data
    # idx1 is assumed to be fixed (like origin)
    pass


#%%        


if __name__=="__main__":
    # scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
    scanfilepath = 'lidarprocessing/houseScan_std.pkl'
    with open(scanfilepath,'rb') as fh:
        dataset=pkl.load(fh)
    
    def getscanpts(dataset,idx):
        # ranges = dataset[i]['ranges']
        rngs = np.array(dataset[idx]['ranges'])
        
        angle_min=dataset[idx]['angle_min']
        angle_max=dataset[idx]['angle_max']
        angle_increment=dataset[idx]['angle_increment']
        ths = np.arange(angle_min,angle_max+angle_increment,angle_increment)
        p=np.vstack([np.cos(ths),np.sin(ths)])
        
        rngidx = (rngs>= dataset[idx]['range_min']) & (rngs<= dataset[idx]['range_max'])
        ptset = rngs.reshape(-1,1)*p.T
        # ptset=np.ascontiguousarray(ptset,dtype=dtype)
        return ptset[rngidx,:]
    X1=getscanpts(dataset,0)
    X2=getscanpts(dataset,50)
    
    Xd,m = get0meanIcov(X1)
    clf,MU,P,W = getclf(Xd)
    sHk_prevframe = np.identity(3)
    
    sHk,err,hess_inv = scan2keyframe_match(clf,m,X2,sHk=sHk_prevframe)
    
    st=time.time()
    sHk,err,hess_inv = scan2keyframe_match(clf,m,X2,sHk=sHk_prevframe)
    et = time.time()
    print("time taken: ",et-st)
    
    x=np.array([np.pi/4,0.1,0.1])
    
    
    