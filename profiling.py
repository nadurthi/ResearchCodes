#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:54:14 2021

@author: na0043
"""

import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.neighbors import KDTree
from uq.gmm import gmmfuncs as uqgmmfnc
from utils.plotting import geometryshapes as utpltgmshp
import time
from scipy.optimize import minimize, rosen, rosen_der,least_squares
from scipy import interpolate
import networkx as nx
import pdb
import pandas as pd
from fastdist import fastdist
import copy
from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point2Dplotting as pt2dplot
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
from sklearn.neighbors import KDTree
import os
import pandas as pd
time_increment = 1.736111516947858e-05
angle_increment = 0.004363323096185923
scan_time = 0.02500000037252903
range_min, range_max = 0.023000000044703484, 60.0
angle_min,angle_max =  -2.3518311977386475,2.3518311977386475

dtype = np.float64

#%%
plt.close("all")
with open("DeutchesMeuseum_g2oTest_good2.pkl",'rb') as fh:
    poseGraph,params,_=pkl.load(fh)

Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))
e1=Lkeyloop_edges[1751][0]
e2=Lkeyloop_edges[1751][1]
X1=poseGraph.nodes[e1]['X']
X2=poseGraph.nodes[e2]['X']
H21=np.identity(3) 
th=0*np.pi/180
R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
H21[0:2,0:2]=R
# H21=poseGraph.edges[e1,e2]['H']
# H21[0:2,2]=H21[0:2,2]+5
H12 = nplinalg.inv(H21)
Lmax=np.array([20,20])
thmax=60*np.pi/180
dxMatch=np.array([0.25,0.25])
dxMax=np.array([5,5])
st=time.time()
# X2small = pt2dproc.binnerDownSampler(X2,dx=0.2,cntThres=1)
Hbin21,cost=pt2dproc.binMatcherAdaptive(X1,X2,H12,Lmax,thmax,dxMatch,dxMax)
et=time.time()
X1=poseGraph.nodes[e1]['X']
X2=poseGraph.nodes[e2]['X']
print("Complete in time = ",et-st)
Hbin12 = nplinalg.inv(Hbin21)
R=Hbin12[0:2,0:2]
t=Hbin12[0:2,2]
X22 = R.dot(X2.T).T+t

plt.figure()
plt.plot(X1[:,0],X1[:,1],'b.')
plt.plot(X22[:,0],X22[:,1],'r.')