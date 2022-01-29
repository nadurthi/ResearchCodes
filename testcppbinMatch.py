#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:43:20 2022

@author: na0043
"""
import os
import pickle as pkl
from pyslam import  slam,binmatch
import json
import time
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import shutil
import copy
# try:
    
# except:
    # print("failed import")
cc=os.getcwd()
p1=Path(os.path.join(cc,"lidarprocessing","__pycache__"))
p2=Path(os.path.join(cc,"lidarprocessing","numba_codes","__pycache__"))
if p1.exists():
    shutil.rmtree(p1)
if p2.exists():
    shutil.rmtree(p2)
# p1.rmdir()
# p2.rmdir()

    
import heapq
import lidarprocessing.numba_codes.binmatchers as binmatchers
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc

dtype=np.float64

from numba.core import types
from numba.typed import Dict
float_2Darray = types.float64[:,:]


plt.close("all")

with open("testBinMatch.pkl","rb") as FF:
    X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase,Hbin12,Hbin21numba,cost0,cost,hh,hR,HLevels,dxs=pkl.load(FF)


# with open("testBinMatch.pkl","wb") as FF:
#     pkl.dump([X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase,Hbin12,Hbin21numba,cost0,cost,hh,hR,HLevels,dxs],FF)



#%%
def binMatcherAdaptive_super(X11,X22,H12,Lmax,thmax,thmin,dxMatch,dxBase):
    #  X11 are global points
    # X22 are points with respect to a local frame (like origin of velodyne)
    # H12 takes points in the velodyne frame to X11 frame (global)
    
    # dxBase=dxMatch*(np.floor(dxBase/dxMatch)+1)
    
    n=histsmudge =2 # how much overlap when computing max over adjacent hist for levels
    H21comp=np.identity(3)
        
    mn=np.zeros(2)
    mx=np.zeros(2)
    mn_orig=np.zeros(2)
    mn_orig[0] = np.min(X11[:,0])
    mn_orig[1] = np.min(X11[:,1])
    
    mn_orig=mn_orig-dxMatch
    
    

    H12mn = H12.copy()
    H12mn[0:2,2]=H12mn[0:2,2]-mn_orig
    X1=X11-mn_orig
    
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    
    t0 = H12mn[0:2,2]
    L0=t0-Lmax
    L1=t0+Lmax
    
    
    
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
    
    
    SolBoxes_init=[]
    for xs in np.arange(0,dxs[0][0],dxs[0][0]):
        for ys in np.arange(0,dxs[0][1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )        
    
    h=[(100000.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)]
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
    
    for j in range(len(thL)):
        th=thL[j]
        R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        XX=np.transpose(R.dot(X22.T))
        XX=np.transpose(H12mn[0:2,0:2].dot(XX.T))#+H12mn[0:2,2]
        Xth[th]=XX
        
        
        for solbox in SolBoxes_init:
            xs,ys,d0,d1 = solbox
            Tj=np.array((d0,d1))
            Oj = np.array((xs,ys)) 
        
            cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
            h.append((-cost2-np.random.rand()/1e10,xs,ys,d0,d1,lvl,th,mxLVL))
        
    heapq.heapify(h)
    
    XX0=np.transpose(H12mn[0:2,0:2].dot(X22.T))+H12mn[0:2,2]
    zz=np.zeros(2,dtype=np.float64)
    cost0=nbpt2Dproc.getPointCost(HLevels[-1],dxs[-1],XX0,zz,dxs[-1])
    
    bb1={'x1':t0[0]-Lmax[0],'y1':t0[1]-Lmax[1],'x2':t0[0]+Lmax[0],'y2':t0[1]+Lmax[1]}
    print("bb1 = ",bb1)
    if dxBase[0]>=0:
        while(1):
            HH=[]
            flg=False
            for i in range(len(h)):
                
                if h[i][3]>dxBase[0] or h[i][4]>dxBase[1]:
                    flg=True
                    (cost,xs,ys,d0,d1,lvl,th,mxLVL)=h[i]
            
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
                            bb2={'x1':xs,'y1':ys,'x2':xs+d0,'y2':ys+d1}
                            x_left = max(bb1['x1'], bb2['x1'])
                            y_top = max(bb1['y1'], bb2['y1'])
                            x_right = min(bb1['x2'], bb2['x2'])
                            y_bottom = min(bb1['y2'], bb2['y2'])
                        
                            if x_right < x_left or y_bottom < y_top:
                                continue
                                 
                            Oj = np.array((xs,ys))
                            cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                            HH.append((-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))
                else:
                    HH.append(h[i])
            h=HH
            if flg==False:
                break
        
        heapq.heapify(h)
    hinit =copy.deepcopy(h)   
    while(1):
        (cost,xs,ys,d0,d1,lvl,th,mxLVL)=heapq.heappop(h)
        mainSolbox = (cost,xs,ys,d0,d1,lvl,th,mxLVL)
        if lvl==mxLVL:
            break
            cnt=0
            for jj in range(len(h)):
                if np.floor(-h[jj][0])==np.floor(-cost) and h[jj][5]<lvl:
                    cnt+=1
            if cnt==0:
                break
            else:
                continue
            
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
                bb2={'x1':xs,'y1':ys,'x2':xs+d0,'y2':ys+d1}
                x_left = max(bb1['x1'], bb2['x1'])
                y_top = max(bb1['y1'], bb2['y1'])
                x_right = min(bb1['x2'], bb2['x2'])
                y_bottom = min(bb1['y2'], bb2['y2'])
            
                if x_right < x_left or y_bottom < y_top:
                    continue
                            
                Oj = np.array((xs,ys))
                cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                heapq.heappush(h,(-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))

    
    t=np.array(mainSolbox[1:3])-t0
    th = mainSolbox[6]
    cost=np.floor(-mainSolbox[0])
    d0,d1=mainSolbox[3:5]
    
    print("H12mn = ",H12mn)
    print("t0 = ",t0)
    
    HcompR=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    HcompR[0:2,0:2]=R
    
    Ht=np.identity(3)
    Ht[0:2,2]=t

    H12comp=np.dot(Ht.dot(H12mn),HcompR)
    H12comp[0:2,2]=H12comp[0:2,2]+mn_orig

    H21comp=nplinalg.inv(H12comp)
    hh=0
    hR=0
    return (H21comp,cost0,cost,hh,hR,HLevels,dxs,Xth,hinit)
    # return H21comp



#%%
D={"icp":{},
   "gicp":{},
   "gicp_cost":{},
   "ndt":{},
   "sig0":0.5,
   "dmax":10,
   "DoIcpCorrection":0}

D["icp"]["enable"]=1
D["icp"]["setMaximumIterations"]=50
D["icp"]["setMaxCorrespondenceDistance"]=25
D["icp"]["setRANSACIterations"]=0.0
D["icp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["icp"]["setTransformationEpsilon"]=1e-3
D["icp"]["setEuclideanFitnessEpsilon"]=1



D["gicp_cost"]["enable"]=0


D["gicp"]["enable"]=0
D["gicp"]["setMaxCorrespondenceDistance"]=20
D["gicp"]["setMaximumIterations"]=50.0
D["gicp"]["setMaximumOptimizerIterations"]=20.0
D["gicp"]["setRANSACIterations"]=0
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

D["DON"]["threshold_curv_lb"]=0.1;
D["DON"]["threshold_curv_ub"]=100000;
D["DON"]["threshold_small_nz_lb"]=-0.5;
D["DON"]["threshold_small_nz_ub"]=0.5;
D["DON"]["threshold_large_nz_lb"]=-5;
D["DON"]["threshold_large_nz_ub"]=5;

D["Lmax"]=list(Lmax.astype(np.float64))
D["thmax"]=thmax
D["thfineres"]=thmin
D["dxMatch"]=list(dxMatch.astype(np.float64))
dxBase=np.array([5,5])
D["dxBase"]=list(dxBase.astype(np.float64))

Hbin21numba
# st=time.time()
# Hbin21numba,cost0,cost,hh,hR,HLevels,dxs,Xth,hinit=binMatcherAdaptive_super(X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase)
# et=time.time()
# print("compute time for numba = ",et-st)

bm=binmatch.BinMatch(json.dumps(D))

bm.computeHlevels(X2Dmap_down)
st=time.time()
sol=bm.getmatch(X1v2D,H12est)
et=time.time()
print("compute time = ",et-st)

print("# of same cost sols = ",len(sol))
print(sol[0].H)
print(sol[0].cost0)
print(sol[0].cost)

print("H12est est = ",H12est)
print("correct Hbin12 = ",Hbin12)
print("correct Hbin21 = ",nplinalg.inv(Hbin12))

for i in range(len(sol)):
    print("---------------------")
    # print(sol[i].H)
    print(sol[i].cost0,sol[i].cost)
    

Hcorr =sol[0].H

Hbin12numba=nplinalg.inv(Hbin21numba)

Xest = H12est[0:2,0:2].dot(X1v2D.T).T+H12est[0:2,2]
Xcorr = Hcorr[0:2,0:2].dot(X1v2D.T).T+Hcorr[0:2,2]
Xcorrnumba = Hbin12numba[0:2,0:2].dot(X1v2D.T).T+Hbin12numba[0:2,2]

figbf = plt.figure("bin-fit")
ax = figbf.add_subplot(111)
ax.cla()
ax.plot(X2Dmap_down[:,0],X2Dmap_down[:,1],'k.')
ax.plot(Xest[:,0],Xest[:,1],'b.',label='est')
ax.plot(Xcorr[:,0],Xcorr[:,1],'r.',label='cpp-corrected')
ax.plot(Xcorrnumba[:,0],Xcorrnumba[:,1],'g.',label='numba-corrected')
ax.legend()
ax.axis("equal")
plt.show()


# Xnumbkeys = np.array([s for s in Xth.keys()])
# Xcppkeys = np.array([s for s in bm.Xth.keys()])

# thvec = Xnumbkeys.round(3)

# Xnumbkeys=np.sort(Xnumbkeys)
# Xcppkeys=np.sort(Xcppkeys)

# import pandas as pd
# hinitmod = [(-int(h[0]),np.round(h[1],3),np.round(h[2],3),np.round(h[3],3),np.round(h[4],3) ,int(h[5]),np.round(h[6],3)) for h in hinit]
# dfhinitmod=pd.DataFrame(hinitmod)

# hinitcpp = [(int(qv.cost),np.round(qv.lb[0],3),np.round(qv.lb[1],3),np.round(qv.dx[0],3),np.round(qv.dx[1],3),int(qv.lvl),np.round(qv.th,3) ) for qv in bm.qvinit]
# dfhinitcpp=pd.DataFrame(hinitcpp)

# d1=dfhinitmod[dfhinitmod[6]==thvec[0]][[0,1,2]].sort_values([0,1,2])
# d2=dfhinitcpp[dfhinitcpp[6]==thvec[0]][[0,1,2]].sort_values([0,1,2])
# print(len(d1),len(d2))


