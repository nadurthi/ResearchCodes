#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:43:20 2022

@author: na0043
"""
import pickle as pkl
from pyslam import  slam,binmatch
import json
import time
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import lidarprocessing.numba_codes.binmatchers as binmatchers

plt.close("all")

with open("testBinMatch.pkl","rb") as FF:
    X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase,Hbin12=pkl.load(FF)

    
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
# dxBase=np.array([5,5])
D["dxBase"]=list(dxBase.astype(np.float64))

st=time.time()
Hbin21numba,cost0,cost,hh,hR=binmatchers.binMatcherAdaptive_super(X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase)
et=time.time()
print("compute time for numba = ",et-st)

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
    print(sol[i].H)
    print(sol[i].cost0)
    print(sol[i].cost)
    

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
