# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:11:58 2020

@author: nadur
"""

from numba import set_num_threads,config, njit, threading_layer,get_num_threads
import numpy as np
# set_num_threads(1)
# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'threadsafe'
config.NUMBA_NUM_THREADS

print("#Threads: %d" % get_num_threads())
print("Threading layer chosen: %s" % threading_layer())

import numpy as np
import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,float32
from numba import njit, prange,jit

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.neighbors import KDTree
# from utils.plotting import geometryshapes as utpltgmshp
import time
from scipy.optimize import minimize, rosen, rosen_der
from scipy import interpolate
from fastdist import fastdist
from uq.gmm import gmmfuncs as uqgmmfnc
import pdb

numba_cache=False
#%% Pose estimation by keyframe
# - manage keyframes
# - estimate pose to closest keyframe

def get0meanIcov(X):
    m=np.mean(X,axis=0)
    Xd=X-m
    return Xd,m

def getclf(X,reg_covar=0.001):
    clf = mixture.GaussianMixture(n_components=20, covariance_type='full',reg_covar=reg_covar)
    clf.fit(X)
    MU=np.ascontiguousarray(clf.means_,dtype=np.float64)
    P=np.ascontiguousarray(clf.covariances_,dtype=np.float64)
    W=np.ascontiguousarray(clf.weights_,dtype=np.float64)
    
    
    return clf,MU,P,W

#%%
@jit(float64(float64[:], float64[:,:], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,cache=True) 
def getcostalign(x,Xt,MU,P,W):
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    p=uqgmmfnc.gmm_eval_fast(Xn,MU,P,W)
    logp = -np.log(p)
    m = np.mean(logp)
    # logp = np.diag(f(Xn[:,0],Xn[:,1]))
    return m

def getgridvec(thset,txset,tyset):
    th,x,y=np.meshgrid(thset,txset,tyset)
    Posegrid = np.vstack([th.ravel(), x.ravel(), y.ravel()]).T
    Posegrid=np.ascontiguousarray(Posegrid,dtype=np.float64)
    return Posegrid
#%%
@jit(float64[:](float64[:,:], float64[:,:], float64[:,:,:],float64[:], float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def gridsearch_alignment(Posegrid,MU,P,W,X):
    
    m=np.zeros(Posegrid.shape[0])
    for i in numba.prange(Posegrid.shape[0]):
        m[i]=getcostalign(Posegrid[i],X,MU,P,W)
    ind = np.argmin(m)
    res = Posegrid[ind]
    return res

#%%
@jit(numba.types.Tuple((float64,float64[:]))(float64[:], float64[:,:], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=True) 
def getcostgradient(x,Xt,MU,P,W):
    npt=Xt.shape[0]
    ncomp=MU.shape[0]
    
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    pcomp=uqgmmfnc.gmm_evalcomp_fast(Xn,MU,P,W)
    p = np.sum(pcomp,axis=1)
    logp = -np.log(p)
    f = np.mean(logp)    
    
    invp = 1/p  # this is a vec with each element for one point in Xt
    dRdth = np.array([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])
    invPP = np.zeros_like(P)
    for i in range(ncomp):
        invPP[i] = nplinalg.inv(P[i])
    z1=np.dot(dRdth,Xt.T).T
    
    dpdth=np.zeros((npt,ncomp),dtype=np.float64)
    dpdx=np.zeros((npt,ncomp),dtype=np.float64)
    dpdy=np.zeros((npt,ncomp),dtype=np.float64)
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

# def getcostgradient_wrapper(x,Xt,MU,P,W):
#     g=getcostgradient(x,Xt,MU,P,W)
#     return g[0],g[1:]

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
def alignscan2keyframe(thset,txset,tyset,MU,P,W,X):
    # get sHk: takes a point from keyframe k to scan frame s
    Posegrid = getgridvec(thset,txset,tyset)
    # pdb.set_trace()
    # st=time.time()
    best=gridsearch_alignment(Posegrid,MU,P,W,X)
    # et=time.time()
    # print("time taken by gridsearch : ",et-st)
    
    # st=time.time()
    # res = minimize(getcostalign, best,args=(X,MU,P,W) ,method='BFGS', tol=1e-5) # 'Nelder-Mead'
    # et=time.time()
    # print("time taken by minimize : ",et-st)
    
    # st=time.time()
    res = minimize(getcostgradient, best,args=(X,MU,P,W),jac=True ,method='BFGS', tol=1e-5) # 'Nelder-Mead'
    # et=time.time()
    # print("time taken by gradient minimize : ",et-st)
    
    # print(res2.fun,res.fun)
    # print(res2.x,res.x)
    
    th=res.x[0]
    t=res.x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=np.float64,order='C')

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=res.fun
    
    return sHk,err

def scan2keyframe_match(KeyFrameClf,m_clf,X,sHk=None):
    # sHrelk is the transformation from k to s
    if sHk is None:
        sHk=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64,order='C')
        thset = np.linspace(-np.pi/3,np.pi/3,10)
        txset = np.linspace(-0.4,0.4,5) # 
        tyset = np.linspace(-0.4,0.4,5) #
    else:
        thset = np.linspace(-np.pi/6,np.pi/6,7)
        txset = np.linspace(-0.2,0.2,5) # 
        tyset = np.linspace(-0.2,0.2,5) #
        
    kHs = nplinalg.inv(sHk)
    Xk=np.matmul(kHs,np.vstack([X.T,np.ones(X.shape[0])])).T  
    Xk=Xk[:,:2]
    Xd=Xk-m_clf
    MU=np.ascontiguousarray(KeyFrameClf.means_,dtype=np.float64)
    P=np.ascontiguousarray(KeyFrameClf.covariances_,dtype=np.float64)
    W=np.ascontiguousarray(KeyFrameClf.weights_,dtype=np.float64)
    Xd = np.ascontiguousarray(Xd,dtype=np.float64) 
    sHk_rel,err = alignscan2keyframe(thset,txset,tyset,MU,P,W,Xd)
    sHk_corrected=np.matmul(sHk,sHk_rel)
    return sHk_corrected,err

def poseGraph_keyFrame_matcher(poseGraph,idx1,idx2):
    # fromidx is idx1, toidx is idx2 
    clf1=poseGraph.nodes[idx1]['clf']
    m_clf1=poseGraph.nodes[idx1]['m_clf']
    sHg_1 = poseGraph.nodes[idx1]['sHg']
    sHg_2 = poseGraph.nodes[idx2]['sHg']
    H21_est = np.matmul(sHg_2,nplinalg.inv(sHg_1))
    X2 = poseGraph.nodes[idx2]['X']
    H21,err21 = scan2keyframe_match(clf1,m_clf1,X2,sHk=H21_est)
    
    return H21,err21

#%% Pose estimation by tree
    
def getcostalign_tree(x,Xt,tree):
    th=x[0]
    t=x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xn=np.matmul(R,Xt.T).T+t
    dist, ind = tree.query(Xn, k=1)  
    d = np.mean(np.mean(dist,axis=1))
    return d

def alignscan2tree(thset,txset,tyset,base_tree,X):
    # get sHk: takes a point from keyframe k to scan frame s
    m=1e10
    best=[thset[0],txset[0],tyset[0]]
    for th in thset:
        for tx in txset:
            for ty in tyset:
                c=getcostalign_tree([th,tx,ty],X,base_tree)
                if c < m:
                    m=c
                    best=[th,tx,ty] 
    
 
    res = minimize(getcostalign_tree, best,args=(X,base_tree) ,method='Nelder-Mead', tol=1e-5)
    th=res.x[0]
    t=res.x[1:]

    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=res.fun
    
    return sHk,err

#%% generic method entry functions
def scan2scan_match(X1,X2,method="GMM",H21=None,skipgrid=False):
    # estimate H21 from 1 to 2
    # method="GMM"
    # method="tree"
    if H21 is None:
        H21 = np.array([[1,0,0],[0,1,0],[0,0,1]])
        thset = np.linspace(-2*np.pi/3,2*np.pi/3,13)
        txset = np.linspace(-0.7,0.7,5) # 
        tyset = np.linspace(-0.7,0.7,5) #
    else:
        thset = np.linspace(-np.pi/4,np.pi/4,7)
        txset = np.linspace(-0.3,0.3,5) # 
        tyset = np.linspace(-0.3,0.3,5) #
    
    if skipgrid:
        thset = [0]
        txset = [0] # 
        tyset = [0] #
        
    H12 = nplinalg.inv(H21)
    X12=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T   
    X12=X12[:,:2]
    
    
    
    if method=='tree':
        tree = KDTree(X1, leaf_size=3)  
        H21_rel,err = alignscan2tree(thset,txset,tyset,tree,X12)
        H21_corrected=np.matmul(H21,H21_rel)
    if method=='GMM':
        clf,MU,P,W=getclf(X1)
        H21_rel,err = alignscan2keyframe(thset,txset,tyset,MU,P,W,X12)
        H21_corrected=np.matmul(H21,H21_rel)
    
    return H21_corrected,err


#%% features
bins=np.arange(0,11,0.25)
db=bins[1]-bins[0]
def get2DptFeat(X):
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
        
@jit(float64(float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=False) 
def globalPoseCost(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    y=np.zeros((int(len(x)/3)+1,3),dtype=np.float64)
    for s in range(int(len(x)/3)):
        y[s+1,:]= x[s*3:s*3+3]
    

    th=y[:,0] # global
    txy=y[:,1:] # global
    F=np.zeros(Hrels.shape[0])

        
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
        
        e=(jHi-jHf.dot(fHi))
        eR=e[0:2,0:2]
        et=e[0:2,2]
        F[idx] = 1*(eR[0,0]**2+eR[1,1]**2) + 50*(et[0]**2 + et[1]**2)
    
    f=np.sum(F)
    # f=1.3
    return f


def adjustPoses(poseGraph,idx1,idx2):
    # adjust the bundle from idx1 to idx2
    # i.e. optimize the global pose matrices, with constraint local poses 
    # idx1 is assumed to be fixed (like origin)
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    Lkey.sort()
    
    Lkey_dict={x:i for i,x in enumerate(Lkey)}
    
    Hrels=[] # Hrels=[[i,j,thji,txji,tyji],...]
    Hrelsidx=[]
    for j in range(1,len(Lkey)):
        idx = Lkey[j]
        for idx_prev in poseGraph.predecessors(idx):
            i = Lkey_dict[idx_prev]
            jHi=poseGraph.edges[idx_prev,idx]['H']
            tji,thji = extractPosAngle(jHi)
            Hrels.append([thji,tji[0],tji[1] ])        
            Hrelsidx.append([i,j])
    Hrels = np.array(Hrels,dtype=np.float64)
    Hrelsidx = np.array(Hrelsidx,dtype=np.int32)
    
    # x0=x0.reshape(-1)
    x0 = np.zeros((len(Lkey)-1,3))            
    firstHg=poseGraph.nodes[Lkey[0]]['sHg']
    gHfirst = nplinalg.inv(firstHg)
    sHg_original={}
    for j in range(1,len(Lkey)):
        idx = Lkey[j]
        sHg=poseGraph.nodes[idx]['sHg']
        sHg_original[idx] = sHg
        
        jHf=np.matmul(sHg,gHfirst) # first to j (local global frame)
        fHj=nplinalg.inv(jHf)
        tj,thj = extractPosAngle(fHj)
        # tj = Hjf[:2,2]  
        x0[j-1,0]=thj
        x0[j-1,1:]=tj
        
    x0=x0.reshape(-1)
    
    st=time.time()
    res = minimize(globalPoseCost, x0,args=(Hrelsidx,Hrels) ,method='BFGS', tol=1e-5)
    et=time.time()
    print(res.fun," time: ",et-st)

    
    x = res.x
    x=x.reshape(-1,3)   
    sHg_updated={}
    for j in range(1,len(Lkey)):     
        thj=x[j-1,0] # global
        tj=x[j-1,1:] # global
    
        fHj=getHmat(thj,tj)
        jHf=nplinalg.inv(fHj)
        sHg_updated[Lkey[j]] = np.matmul(jHf,firstHg)
        
    return sHg_updated,sHg_original

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
        # ptset=np.ascontiguousarray(ptset,dtype=np.float64)
        return ptset[rngidx,:]
    X1=getscanpts(dataset,0)
    X2=getscanpts(dataset,50)
    
    Xd,m = get0meanIcov(X1)
    clf,MU,P,W = getclf(Xd)
    sHk_prevframe = np.identity(3)
    
    sHk,err = scan2keyframe_match(clf,m,X2,sHk=sHk_prevframe)
    
    st=time.time()
    sHk,err = scan2keyframe_match(clf,m,X2,sHk=sHk_prevframe)
    et = time.time()
    print("time taken: ",et-st)
    
    x=np.array([np.pi/4,0.1,0.1])
    
    
    