#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 09:42:08 2022

@author: na0043
"""
import numpy as np
import numpy.linalg as nplinalg
from pyslam import slam 
import json
import open3d as o3d
import time 
import pickle
from joblib import dump, load
# from lidarprocessing.numba_codes import gicp


#%%
basedir ='/media/na0043/misc/DATA/KITTI/odometry/dataset'
# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
loop_closed_seq = ['02','05','06','08']
sequence = '05'

#%%

with open("testpcllocalize.pkl","rb") as FF:
    Xpf,X1v_down=pickle.load(FF)

pcd=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)
pcdboxdown=pcd.voxel_down_sample(voxel_size=0.2)

# bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(X1v_down,axis=0)-50,max_bound=np.max(X1v_down,axis=0)+50)
# pcdboxdown=pcd.voxel_down_sample(voxel_size=0.1).crop(bbox3d)

D={"icp":{},
   "gicp":{},
   "gicp_cost":{},
   "ndt":{},
   "sig0":0.5,
   "dmax":50}

D["icp"]["enable"]=0
D["icp"]["setMaximumIterations"]=5
D["icp"]["setMaxCorrespondenceDistance"]=25
D["icp"]["setRANSACIterations"]=0.0
D["icp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["icp"]["setTransformationEpsilon"]=1e-3
D["icp"]["setEuclideanFitnessEpsilon"]=1


D["gicp_cost"]["enable"]=0


D["gicp"]["enable"]=1
D["gicp"]["setMaxCorrespondenceDistance"]=20
D["gicp"]["setMaximumIterations"]=10.0
D["gicp"]["setMaximumOptimizerIterations"]=10.0
D["gicp"]["setRANSACIterations"]=0.0
D["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["gicp"]["setTransformationEpsilon"]=1e-9
D["gicp"]["setUseReciprocalCorrespondences"]=1

D["ndt"]["enable"]=0
D["ndt"]["setTransformationEpsilon"]=1e-9
D["ndt"]["setStepSize"]=2.0
D["ndt"]["setResolution"]=1.0
D["ndt"]["setMaximumIterations"]=25.0
D["ndt"]["initialguess_axisangleA"]=0.0
D["ndt"]["initialguess_axisangleX"]=0.0
D["ndt"]["initialguess_axisangleY"]=0.0
D["ndt"]["initialguess_axisangleZ"]=1.0
D["ndt"]["initialguess_transX"]=0.5
D["ndt"]["initialguess_transY"]=0.01
D["ndt"]["initialguess_transZ"]=0.01

D["DON"]={}
D["DON"]["scale1"]=1;
D["DON"]["scale2"]=2;
D["DON"]["threshold"]=0.2;
D["DON"]["threshold_small_z"]=0.5;
D["DON"]["threshold_large_z"]=0.5;

D["DON"]["segradius"]=1;

opts = json.dumps(D)

pcldon = slam.Don(opts)


X=np.asarray(pcdboxdown.points)
st=time.time()
pcldon.setMapX(X)
et=time.time()
print("time taken for DON map load = ",et-st)

st=time.time()
pcldon.computeNormals(opts)
pcldon.computeDon(opts)
et=time.time()
print("time taken for DON map load = ",et-st)

#
# D["DON"]["threshold"]=0.5;
# D["DON"]["threshold_small_z"]=0.5;
# D["DON"]["threshold_large_z"]=0.5;

D["DON"]["threshold_curv_lb"]=0.1;
D["DON"]["threshold_curv_ub"]=100000;
D["DON"]["threshold_small_nz_lb"]=-0.5;
D["DON"]["threshold_small_nz_ub"]=0.5;
D["DON"]["threshold_large_nz_lb"]=-5;
D["DON"]["threshold_large_nz_ub"]=5;

opts = json.dumps(D)
st=time.time()
ret=pcldon.filter(opts)
ret=dict(ret)
Xout=ret["Xout"]
et=time.time()
print("time taken for DON map load = ",et-st)

# C=nplinalg.norm(Xout[:,3:6]-Xout[:,6:9],axis=1)

# idx=(C>0.2) & (np.abs(Xout[:,5])<0.5)
print(Xout.shape)

pcdout = o3d.geometry.PointCloud()
pcdout.points = o3d.utility.Vector3dVector(Xout[:,:3])
# pcdout.paint_uniform_color([0,1,0]) #green
o3d.visualization.draw_geometries([pcdout])
o3d.io.write_point_cloud("kitti-pcd-seq-roadremove-%s.pcd"%sequence, pcdout)


#%%
# X1=Xout[:,:2]





pcd=o3d.io.read_point_cloud("kitti-pcd-seq-roadremove-%s.pcd"%sequence)
np.asarray(pcd.points)[:,2]=0
o3d.visualization.draw_geometries([pcd])


import matplotlib.pyplot as plt
XX=np.array(pcd.voxel_down_sample(voxel_size=0.1).points)

XX=XX[:,:2]


from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point2Dplotting as pt2dplot
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
from uq.gmm import gmmfuncs as uqgmmfnc
from utils.plotting import geometryshapes as utpltgmshp

params={}
params['n_components']=500
params['reg_covar']=0.002

res = pt2dproc.getclf(XX,params,doReWtopt=True,means_init=None)
clf=res['clf']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(XX[:,0],XX[:,1],'k.')

for i in range(clf.n_components):
    m = clf.means_[i]
    P = clf.covariances_[i]
    Xe= utpltgmshp.getCovEllipsePoints2D(m,P,nsig=2,N=100)
    
    ax.plot(Xe[:,0],Xe[:,1],'r')
    
ax.axis('equal')
plt.show()


#%% binmatch
X11=XX

dxMatch=np.array([0.25,0.25])


mn=np.zeros(2)
mx=np.zeros(2)
mn_orig=np.zeros(2)
mn_orig[0] = np.min(X11[:,0])
mn_orig[1] = np.min(X11[:,1])

mn_orig=mn_orig-dxMatch



X1=X11-mn_orig

# print("mn_orig = ",mn_orig)

mn[0] = np.min(X1[:,0])
mn[1] = np.min(X1[:,1])
mx[0] = np.max(X1[:,0])
mx[1] = np.max(X1[:,1])


# print("mn,mx=",mn,mx)
P = mx-mn




# dxMax[0] = np.min([dxMax[0],Lmax[0]/2,P[0]/2])
# dxMax[1] = np.min([dxMax[1],Lmax[1]/2,P[1]/2])

mxlvl=0
dx0=mx+dxMatch
dxs = []
XYedges=[]
for i in range(0,100):
    f=2**i
    
    xedges=np.linspace(0,mx[0]+1*dxMatch[0],f+1)
    yedges=np.linspace(0,mx[1]+1*dxMatch[0],f+1)
    XYedges.append((xedges,yedges))
    dx=np.array([xedges[1]-xedges[0],yedges[1]-yedges[0]])

    dxs.append(dx)
    
    if np.any(dx<=dxMatch):
        break
    
mxlvl=len(dxs)

# dxs=dxs[::-1]
dxs=[dx.astype(np.float32) for dx in dxs]
# XYedges=XYedges[::-1]


H1match=nbpt2Dproc.numba_histogram2D(X1, XYedges[-1][0],XYedges[-1][1])
H1match[H1match>0]=1




# first create multilevel histograms
levels=[]
HLevels=[H1match]




n=2
for i in range(1,mxlvl):
    


    Hup = HLevels[i-1]
    # H=pool2d(Hup, kernel_size=3, stride=2, padding=1, pool_mode='max')
    H=nbpt2Dproc.UpsampleMax(Hup,n)
    


    # pt2dproc.plotbins2(XYedges[i][0],XYedges[i][1],H,X1,X2)

    HLevels.append(H)
      

mxLVL=len(HLevels)-1
HLevels=HLevels[::-1]
HLevels=[np.ascontiguousarray(H).astype(np.int32) for H in HLevels]

for i in range(mxlvl):
    H=HLevels[i]
    pt2dproc.plotbins2(XYedges[i][0],XYedges[i][1],H,X1,X1)



#%% Real bin matching
plt.close("all")
import heapq
numba_cache=True
dtype=np.float64

from numba.core import types
from numba.typed import Dict
float_2Darray = types.float64[:,:]

def binMatcherAdaptive3(X11,X22,H12,Lmax,thmax,thmin,dxMatch):
    #  X11 are global points
    # X22 are points with respect to a local frame (like origin of velodyne)
    # H12 takes points in the velodyne frame to X11 frame (global)
    
    n=histsmudge =2 # how much overlap when computing max over adjacent hist for levels
    H21comp=np.identity(3)
        
    mn=np.zeros(2)
    mx=np.zeros(2)
    mn_orig=np.zeros(2)
    mn_orig[0] = np.min(X11[:,0])
    mn_orig[1] = np.min(X11[:,1])
    
    mn_orig=mn_orig-dxMatch
    
    
    # R=H12[0:2,0:2]
    # t=H12[0:2,2]
    # X222 = R.dot(X22.T).T+t
    
    H12mn = H12.copy()
    H12mn[0:2,2]=H12mn[0:2,2]-mn_orig
    
    # X2=X222-mn_orig
    X1=X11-mn_orig
    
    
    # t2 = np.mean(X2,axis=0)
    # X20 = X2-t2
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    # rmax=np.max(np.sqrt(X2[:,0]**2+X2[:,1]**2))

    P = mx-mn

    mxlvl=0
    dx0=mx+dxMatch
    dxs = []
    XYedges=[]
    for i in range(0,100):
        f=2**i
        
        xedges=np.linspace(0,mx[0]+1*dxMatch[0],f+1)
        yedges=np.linspace(0,mx[1]+1*dxMatch[0],f+1)
        XYedges.append((xedges,yedges))
        dx=np.array([xedges[1]-xedges[0],yedges[1]-yedges[0]])
    
        dxs.append(dx)
        
        if np.any(dx<=dxMatch):
            break
        
    mxlvl=len(dxs)
    

    dxs=[dx.astype(np.float64) for dx in dxs]

    
    
    H1match=nbpt2Dproc.numba_histogram2D(X1, XYedges[-1][0],XYedges[-1][1])
    H1match = np.sign(H1match)
    
    
    
    

    levels=[]
    HLevels=[H1match]
    
    
    
    
    
    for i in range(1,mxlvl):
        
    
    
        Hup = HLevels[i-1]
        H=nbpt2Dproc.UpsampleMax(Hup,n)

        HLevels.append(H)
          
    
    mxLVL=len(HLevels)-1
    HLevels=HLevels[::-1]
    HLevels=[np.ascontiguousarray(H).astype(np.int32) for H in HLevels]
    
    # P2=0.001*P
    # Lmax=P2*(np.floor(Lmax/P2)+1)
    # Lmax=np.maximum(Lmax,P)
    
    LmaxOrig=np.zeros(2,dtype=np.float64)
    LmaxOrig[0]=Lmax[0]
    LmaxOrig[1]=Lmax[1]

    SolBoxes_init=[]
    # X2[:,0]=X2[:,0]-LmaxOrig[0]
    # X2[:,1]=X2[:,1]-LmaxOrig[1]
    
    # print(Lmax)
    # Lmax=dxs[-1]*(np.floor(Lmax/dxs[-1])+1)
    # print(Lmax)
    for xs in np.arange(-Lmax[0],Lmax[0],dxs[0][0]):
        for ys in np.arange(-Lmax[1],Lmax[1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )
    
    
    
    
    
    
    h=[(100000.0,0.0,0.0,0.0,0.0,0.0,0.0)]
    lvl=0
    dx=dxs[lvl]
    H=HLevels[lvl]
    
    Xth= Dict.empty(
        key_type=types.float64,
        value_type=float_2Darray,
    )
    ii=0

    thfineRes = thmin
    thL=np.arange(-thmax,thmax+thfineRes,thfineRes,dtype=np.float64)
    
    
    # X2=np.ascontiguousarray(X2)
    for j in range(len(thL)):
        th=thL[j]
        R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        XX=np.transpose(R.dot(X22.T))
        XX=np.transpose(H12mn[0:2,0:2].dot(XX.T))+H12mn[0:2,2]
        Xth[th]=XX
        
        
        for solbox in SolBoxes_init:
            xs,ys,d0,d1 = solbox
            Tj=np.array((d0,d1))
            Oj = np.array((xs,ys))
            cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
            h.append((-cost2-np.random.rand()/1e10,xs,ys,d0,d1,lvl,th))
            
    heapq.heapify(h)
    
    # print("len(heap) = ",len(h))

    while(1):
        (cost,xs,ys,d0,d1,lvl,th)=heapq.heappop(h)
        mainSolbox = (cost,xs,ys,d0,d1,lvl,th)
        if lvl==mxLVL:
            break
        
        nlvl = int(lvl)+1
        dx=dxs[nlvl]
        H=HLevels[nlvl]
        Tj=np.array((d0,d1))
        Oj = np.array((xs,ys))

        
        
        Xg=np.arange(Oj[0],Oj[0]+Tj[0],dx[0])
        Yg=np.arange(Oj[1],Oj[1]+Tj[1],dx[1])
        
        d0,d1=dx[0],dx[1]
        Tj=np.array((d0,d1))
        
        for xs in Xg[:2]:
            for ys in Yg[:2]:
                Oj = np.array((xs,ys))
                cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                heapq.heappush(h,(-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th))

    t=mainSolbox[1:3]
    th = mainSolbox[6]
    cost=-mainSolbox[0]
    

    
    HcompR=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    HcompR[0:2,0:2]=R
    
    Ht=np.identity(3)
    Ht[0:2,2]=t
    
    # first rotation by HcompR or R
    # then H12mn
    # then translate by t
    # Ht*H12mn*HcompR*X22v
    H12comp=np.dot(Ht.dot(H12mn),HcompR)
    H12comp[0:2,2]=H12comp[0:2,2]+mn_orig
    
    

    
    # H1=np.identity(3)
    # H1[0:2,2]=-mn_orig
    
    # Hlmax=np.identity(3)
    # Hlmax[0:2,2]=-LmaxOrig

    
    # H2=nplinalg.inv(H1)
    # H2=H2.dot(Hcomp)
    # H3=H2.dot(Hlmax)
    # H4=H3.dot(H1)
    # H12comp=H4.dot(H12)

    H21comp=nplinalg.inv(H12comp)
    
    return H21comp
    

dxMatch=np.array([2,2])

def down_sample(X,voxel_size):
    NN=X.ndim
    if NN==2:
        X=np.hstack([X,np.zeros((X.shape[0],1))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    XX=np.asarray(voxel_down_pcd.points)
    return XX[:,:NN]
    
idx=(X11[:,0]>50) & (X11[:,0]<125) & (X11[:,1]>-50) & (X11[:,1]<50)

X2=X11[idx,:]
X2=down_sample(X2,dxMatch[0])

tm=np.mean(X2,axis=0)
X2=X2-tm
H12est=np.identity(3)
H12est[0:2,2]=tm+25

th=-15*np.pi/180
R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
X2=R.dot(X2.T).T

X2g=H12est[0:2,0:2].dot(X2.T).T+H12est[0:2,2]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X11[:,0],X11[:,1],'k.')
ax.plot(X2g[:,0],X2g[:,1],'b.')



Lmax=np.array([30,30])
thmax=20*np.pi/180
thmin=5*np.pi/180
st=time.time()
# Hbin21=binMatcherAdaptive3(X11,X2,H12est,Lmax,thmax,thmin,dxMatch)
Hbin21=nbpt2Dproc.binMatcherAdaptive4_good(X11,X2,H12est,Lmax,thmax,thmin,dxMatch)
et=time.time()
print("time for bin match = ",et-st)
# nbpt2Dproc.

Hbin12=nplinalg.inv(Hbin21) 
X2t=Hbin12[0:2,0:2].dot(X2.T).T+Hbin12[:2,2]

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(X11[:,0],X11[:,1],'k.')
ax.plot(X2t[:,0],X2t[:,1],'r.')

