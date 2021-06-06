#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 18:17:16 2021

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
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point2Dplotting as pt2dplot
from sklearn.neighbors import KDTree
import pandas as pd
time_increment = 1.736111516947858e-05
angle_increment = 0.004363323096185923
scan_time = 0.02500000037252903
range_min, range_max = 0.023000000044703484, 60.0
angle_min,angle_max =  -2.3518311977386475,2.3518311977386475

dtype = np.float64

#%%

with open("PoseGraph-deutchesMesuemDebug-planes",'rb') as fh:
    poseGraph,=pkl.load(fh)
    

plt.close("all")
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))

previdx= Lkey[0]
idx = Lkey[11]

params={}
params['doLoopClosure'] = True
params['LOOP_CLOSURE_D_THES']=0.29
params['LOOP_CLOSURE_POS_THES']=200
params['LOOP_CLOSURE_ERR_THES']= 3
params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
params['LOOPCLOSE_BIN_MATCHER_L'] = 15


params['xy_hess_inv_thres']=10000*0.4
params['th_hess_inv_thres']=10000*0.4
params['#ThreadsLoopClose']=8

st = time.time()
piHi,pi_err_i,hess_inv_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,idx,previdx,params)
et = time.time()
print("time taken = ",et-st)
h1=poseGraph.nodes[idx]['h']
h2=poseGraph.nodes[previdx]['h']
p1=poseGraph.nodes[idx]['pos']
p2=poseGraph.nodes[previdx]['pos']
d=nplinalg.norm(h1-h2,ord=1)

pt2dplot.plotcomparisons(poseGraph,idx,previdx,H12=nplinalg.inv(piHi),err=pi_err_i) #nplinalg.inv(piHi) 

print("d,LOOP_CLOSURE_D_THES = ",d,params['LOOP_CLOSURE_D_THES'])