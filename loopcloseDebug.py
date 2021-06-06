# -*- coding: utf-8 -*-
"""
Created on Sat May 22 02:48:18 2021

@author: Nagnanamus
"""
import os
os.environ["OPENBLAS_MAIN_FREE"] = "1"

import multiprocessing as mp
import threading
import queue

import sys
# sys.path.append(r'/home/na0043/Insync/n.adurthi@gmail.com/Google Drive/repos/SLAM')

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
from sklearn.neighbors import KDTree
import pandas as pd
time_increment = 1.736111516947858e-05
angle_increment = 0.004363323096185923
scan_time = 0.02500000037252903
range_min, range_max = 0.023000000044703484, 60.0
angle_min,angle_max =  -2.3518311977386475,2.3518311977386475

dtype = np.float64




#%%

params={}

params['doLoopClosure'] = True
params['LOOP_CLOSURE_D_THES']=0.29
params['LOOP_CLOSURE_POS_THES']=20
params['LOOP_CLOSURE_ERR_THES']= 3
params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
params['LOOPCLOSE_BIN_MATCHER_L'] = 15


params['xy_hess_inv_thres']=10000*0.4
params['th_hess_inv_thres']=10000*0.4
params['#ThreadsLoopClose']=4
 
    
if __name__ == '__main__':
    with open("PoseGraph-deutchesMesuemDebug-planes",'rb') as fh:
        poseGraph,=pkl.load(fh)
    
    Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    
    plt.close("all")
    ee=[]
    E = list(poseGraph.edges)
    for e in E:
        if 'edgetype' in poseGraph.edges[e[0],e[1]]:
            if poseGraph.edges[e[0],e[1]]["edgetype"]=="Key2Key-LoopClosure":
                ee.append(e)
    for e in ee:
        poseGraph.remove_edge(*e)
    
    poseData={}
    for idx in poseGraph.nodes:
        poseData[idx]={'X':poseGraph.nodes[idx]['X']}
        poseGraph.nodes[idx]['LoopDetectDone'] = False
        
            
    poseGraph=pt2dproc.detectAllLoopClosures(poseGraph,poseData,params,returnCopy=False)
    
    
    with open("PoseGraph-deutchesMesuemDebug-planes-loopclosed",'wb') as fh:
        pkl.dump([poseGraph],fh)
        
    # pt2dplot.plot_keyscan_path(poseGraph,Lkeys[0],Lkeys[-1],makeNew=False,skipScanFrame=True,plotGraphbool=True,
    #                                    forcePlotLastidx=True,plotLastkeyClf=True)
    # plt.show()
    
    # res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,Lkey[0],Lkey[1500])
    # if res.success:
    #     poseGraph2=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated)
    #     # poseGraph = copy.deepcopy(poseGraph2)
    #     pt2dplot.plot_keyscan_path(poseGraph2,Lkey[0],Lkey[-1],makeNew=True,skipScanFrame=True,plotGraphbool=True,
    #                                    forcePlotLastidx=True,plotLastkeyClf=True)
        
    # else:
    #     print("loop close opt failed")
    