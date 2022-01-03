# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 19:55:35 2021

@author: Nagnanamus
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
import copy
from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point2Dplotting as pt2dplot
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
from sklearn.neighbors import KDTree
import os
import pandas as pd
import heapq
import numpy.linalg as nplinalg 
from lidarprocessing import icp

dtype = np.float64
#%%
plt.close("all")
with open("DeutchesMeuseum_g2oTest_good2.pkl",'rb') as fh:
    poseGraph,params,_=pkl.load(fh)
    
    

Lmax=np.array([10,10])
thmax=35*np.pi/180
dxMatch=np.array([0.1,0.1])
H12est = np.identity(3) 
thmin=2*np.pi/180





Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))

# select idex for the edges
idex=1
e1=Lkeyloop_edges[idex][0]
e2=Lkeyloop_edges[idex][1]
X1=poseGraph.nodes[e1]['X']
X2=poseGraph.nodes[e2]['X']

# the bin match is more robust if the downsample bin size is same as the match size.
X1=pt2dproc.binnerDownSampler(X1,dx=dxMatch[0],cntThres=1)
X2=pt2dproc.binnerDownSampler(X2,dx=dxMatch[0],cntThres=1)

# H21gmm was computed using the Gaussian mixture using EM algorithm
H21gmm = poseGraph.edges[e1,e2]['H']

# H21=poseGraph.edges[e1,e2]['H']
# H21[0:2,2]=H21[0:2,2]+5
# H12 = nplinalg.inv(H21)
# X2=R.dot(X1.T).T+H21[0:2,2]

plt.figure()
plt.plot(X1[:,0],X1[:,1],'b.')
plt.plot(X2[:,0],X2[:,1],'r.')


st=time.time()
H12est = np.identity(3) 
Hbin21=nbpt2Dproc.binMatcherAdaptive3(X1,X2,H12est,Lmax,thmax,thmin,dxMatch)
et=time.time()

print("Complete in time = ",et-st)
Hbin12 = nplinalg.inv(Hbin21)
R=Hbin12[0:2,0:2]
t=Hbin12[0:2,2]
X22 = R.dot(X2.T).T+t

plt.figure()
plt.plot(X1[:,0],X1[:,1],'b.')
plt.plot(X22[:,0],X22[:,1],'r.')
plt.title("Binmatch")
plt.show()


tgmm,thgmm=nbpt2Dproc.extractPosAngle(H21gmm)
tes,thes=nbpt2Dproc.extractPosAngle(Hbin21)
tdiff=nplinalg.norm(tgmm-tes)
thdiff=np.abs(thgmm-thes)
print("Binmatch error = ",tdiff,thdiff)


# now run the ICP
Hicp21, distances, i=icp.icp(X1, X2, init_pose=None, max_iterations=500, tolerance=0.01)

Hicp12 = nplinalg.inv(Hicp21)
R=Hicp12[0:2,0:2]
t=Hicp12[0:2,2]
X22 = R.dot(X2.T).T+t

plt.figure()
plt.plot(X1[:,0],X1[:,1],'b.')
plt.plot(X22[:,0],X22[:,1],'r.')
plt.title("ICP")
plt.show()


tgmm,thgmm=nbpt2Dproc.extractPosAngle(H21gmm)
ticp,thicp=nbpt2Dproc.extractPosAngle(Hicp21)
tdiff=nplinalg.norm(tgmm-ticp)
thdiff=np.abs(thgmm-thicp)
print("ICP error = ",tdiff,thdiff)

