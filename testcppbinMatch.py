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

with open("testBinMatch.pkl","rb") as FF:
    X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase=pkl.load(FF)
    
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
D["dxBase"]=list(dxBase.astype(np.float64))

bm=binmatch.BinMatch(json.dumps(D))

bm.computeHlevels(X2Dmap_down)