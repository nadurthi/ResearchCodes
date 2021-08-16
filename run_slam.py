#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:35:09 2021

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

# import warnings
# warnings.filterwarnings('error')

 #%%
# scanfilepath = 'C:/Users/nadur/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
# scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_complete.pkl'
# scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
# scanfilepath = 'lidarprocessing/datasets/DeutchMeuseum/b2-2015-07-07-11-27-05.pkl'
# scanfilefolder = 'lidarprocessing/datasets/DeutchMeuseum/b2-2015-07-07-11-27-05/'
# scanfilepath = 'lidarprocessing/datasets/DeutchMeuseum/b2-2014-11-24-14-33-46.pkl'
scanfilefolder = 'lidarprocessing/datasets/DeutchMeuseum/b2-2014-11-24-14-33-46/'
# os.makedirs(scanfilefolder)

# with open(scanfilepath,'rb') as fh:
#     dataset=pkl.load(fh)


# for i in range(len(dataset['scan'])):
#     with open(os.path.join(scanfilefolder,'scan_%d.pkl'%i),'wb') as F:
#         pkl.dump(dataset['scan'][i],F)

def getscanpts_deutches(idx):
    
    with open(os.path.join(scanfilefolder,'scan_%d.pkl'%idx),'rb') as F:
        scan=pkl.load(F)
    rngs = list(map(lambda x: np.max(x) if len(x)>0 else 120,scan['scan']))
    rngs = np.array(rngs)
    
    # ranges = dataset[i]['ranges']
    # rngs = np.array(dataset[idx]['ranges'])
    
    ths = np.arange(angle_min,angle_max,angle_increment)
    p=np.vstack([np.cos(ths),np.sin(ths)])
    
    rngidx = (rngs> (range_min+0.1) ) & (rngs< (range_max-5))
    ptset = rngs.reshape(-1,1)*p.T
    
    X=ptset[rngidx,:]
    
    Xd=pt2dproc.binnerDownSampler(X,dx=0.025,cntThres=1)
                
    # now filter silly points
    tree = KDTree(Xd, leaf_size=5)
    cnt = tree.query_radius(Xd, 0.25,count_only=True) 
    Xd= Xd[cnt>=2,:]
    
    cnt = tree.query_radius(Xd, 0.5,count_only=True) 
    Xd = Xd[cnt>=5,:]
    
    return Xd


# class IntelData:
#     def __init__(self):
#         intelfile = "lidarprocessing/datasets/Intel Research Lab.clf"
#         ff=open(intelfile)
#         self.inteldata=ff.readlines()
#         self.inteldata = list(filter(lambda x: '#' not in x,self.inteldata))
#         self.flaserdata = list(filter(lambda x: 'FLASER' in x[:8],self.inteldata))
#         self.odomdata = list(filter(lambda x: 'ODOM' in x[:8],self.inteldata))
        
#     def __len__(self):
#         return len(self.flaserdata)
    
#     def getflaser(self,idx):
#         g = self.flaserdata[idx]
#         glist = g.strip().split(' ')
#         # print(glist)
#         cnt = int(glist[1])
#         rngs =[]
#         ths=np.arange(0,90*np.pi/180,0.5*np.pi/180)
#         for i in range(cnt):
#             rngs.append(float(glist[2+i]))
#         rngs = np.array(rngs)
#         p=np.vstack([np.cos(ths),np.sin(ths)])
#         ptset = rngs.reshape(-1,1)*p.T
        
#         return ptset
# dataset = IntelData()       
# def getscanpts_intel(dataset,idx):
#     # ranges = dataset[i]['ranges']
   
#     X=dataset.getflaser(idx)
    
#     # now filter silly points
#     tree = KDTree(X, leaf_size=5)
#     cnt = tree.query_radius(X, 0.25,count_only=True) 
#     X = X[cnt>=2,:]
    
#     cnt = tree.query_radius(X, 0.5,count_only=True) 
#     X = X[cnt>=5,:]
    
#     return X

getscanpts = getscanpts_deutches
#%% TEST::::::: Pose estimation by keyframe
plt.close("all")
poses=[]
poseGraph = nx.DiGraph()


# Xr=np.zeros((len(dataset),3))
ri=0
KeyFrames=[]

params={}

params['REL_POS_THRESH']=0.5 # meters after which a keyframe is made
params['REL_ANGLE_THRESH']=15*np.pi/180
params['ERR_THRES']=4
params['n_components']=35
params['reg_covar']=0.002

params['BinDownSampleKeyFrame_dx']=0.05
params['BinDownSampleKeyFrame_probs']=0.1

params['Plot_BinDownSampleKeyFrame_dx']=0.05
params['Plot_BinDownSampleKeyFrame_probs']=0.001

params['doLoopClosure'] = True
params['Loop_CLOSURE_PARALLEL'] = True
params['LOOP_CLOSURE_D_THES']=31.4
params['LOOP_CLOSURE_POS_THES']=25
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
# params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
# params['LOOPCLOSE_BIN_MATCHER_L'] = 13
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = np.array([0.25,0.25],dtype=np.float64)
params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.2
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']=0.7
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']=0.5
params['LOOP_CLOSURE_COMBINE_MAX_NODES']= 16
params['offsetNodesBy'] = 2

params['NearLoopClose'] = {}
params['NearLoopClose']['Method']='GMM'
params['NearLoopClose']['PoseGrid']=None
params['NearLoopClose']['isPoseGridOffset']=True
params['NearLoopClose']['isBruteForce']=False


# meters. skip loop closure of current node if there is a loop closed node within radius along the path
params['LongLoopClose'] = {}
params['LongLoopClose']['Method'] = 'GMM'
params['LongLoopClose']['SkipLoopCloseIfNearCLosedNodeWithin'] = 5 
params['LongLoopClose']['PoseGrid']=None #pt2dprocgetgridvec(np.linspace(),txset,tyset)
params['LongLoopClose']['isPoseGridOffset']=True
params['LongLoopClose']['isBruteForce']=False

# params['Do_GMM_FINE_FIT']=False

# params['Do_BIN_FINE_FIT'] = False

params['Do_BIN_DEBUG_PLOT-dx']=False
params['Do_BIN_DEBUG_PLOT']= False

params['xy_hess_inv_thres']=100000000*0.4
params['th_hess_inv_thres']=100000000*0.4
params['#ThreadsLoopClose']=8

params['INTER_DISTANCE_BINS_max']=120
params['INTER_DISTANCE_BINS_dx']=1




DoneLoops=[]
# fig = plt.figure("Full Plot")
# ax = fig.add_subplot(111)

# figg = plt.figure("Graph Plot")
# axgraph = figg.add_subplot(111)

Nframes = len(os.listdir(scanfilefolder))
# Nframes = len(dataset)

idx1=15000 #16000 #27103 #14340
idxLast = Nframes
previdx_loopclosure = idx1
previdx_loopdetect = idx1

for idx in range(idx1,idxLast): 
    # ax.cla()
    X=getscanpts(idx)
    if len(poseGraph.nodes)==0:
        # Xd,m = pt2dproc.get0meanIcov(X)
        res = pt2dproc.getclf(X,params,doReWtopt=True,means_init=None)
        clf=res['clf']
        H=np.hstack([np.identity(2),np.zeros((2,1))])
        H=np.vstack([H,[0,0,1]])
        idbmx = params['INTER_DISTANCE_BINS_max']
        idbdx=params['INTER_DISTANCE_BINS_dx']
        # h=pt2dproc.get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
        h = np.array([0,0])
        poseGraph.add_node(idx,frametype="keyframe",X=X,clf=clf,time=idx,sHg=H,pos=(0,0),h=h,color='g',LoopDetectDone=False)
        # poseData[idx]={'X':X}
        
        KeyFrame_prevIdx=idx
        KeyFrames.append(np.array([0,0,0]))
        continue
    
    # estimate pose to last keyframe
    KeyFrameClf = poseGraph.nodes[KeyFrame_prevIdx]['clf']
    Xclf = poseGraph.nodes[KeyFrame_prevIdx]['X']
    # m_clf = poseGraph.nodes[KeyFrame_prevIdx]['m_clf']
    if (idx-KeyFrame_prevIdx)<=1:
        sHk_prevframe = np.identity(3)
    else:
        sHk_prevframe = poseGraph.edges[KeyFrame_prevIdx,idx-1]['H']
    # assuming sHk_prevframe is very close to sHk
    st=time.time()
    sHk,serrk,shessk_inv = pt2dproc.scan2keyframe_match(KeyFrameClf,Xclf,X,params,sHk=sHk_prevframe)
    # sHk,serrk,shessk_inv = pt2dproc.scan2keyframe_match(KeyFrameClf,X,sHk=sHk_prevframe)
    # shessk_inv is like covariance
    et = time.time()
    
    # if np.isfinite(serrk)==0:
    #     pdb.set_trace()
    #     pt2dplot.plotcomparisons_posegraph(poseGraph,KeyFrame_prevIdx,idx,H12=nplinalg.inv( sHk_prevframe) )
        
        
    print("idx = ",idx," Error = ",np.round(serrk,5)," , and time taken = ",np.round(et-st,3))
    # publish pose
    kHg = poseGraph.nodes[KeyFrame_prevIdx]['sHg']
    sHg = np.matmul(sHk,kHg)
    poses.append(sHg)
    gHs=nplinalg.inv(sHg)
    
    # get relative frame from idx-1 to idx
    # iHim1 = np.matmul(sHk,nplinalg.inv(sHk_prevframe))
    tprevK,thprevK = nbpt2Dproc.extractPosAngle(kHg)
    tcurr,thcurr = nbpt2Dproc.extractPosAngle(sHg)
    thdiff = np.abs(nbpt2Dproc.anglediff(thprevK,thcurr))
    # print("thdiff = ",thdiff)
    # check if to make this the keyframe
    if (serrk>params['ERR_THRES'] or nplinalg.norm(sHk[:2,2])>params['REL_POS_THRESH'] or (idx-KeyFrame_prevIdx)>100) or thdiff>params['REL_ANGLE_THRESH']:
        print("New Keyframe")
        st = time.time()
        pt2dproc.addNewKeyFrame(poseGraph,X,idx,KeyFrame_prevIdx,sHg,params,keepOtherScans=False)
        et=time.time()
        print("time taken for new keyframe = ",et-st)
        poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,H_prevframe=sHk_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Key",color='k')
    
    
        KeyFrame_prevIdx = idx
        KeyFrames.append(np.matmul(gHs,np.array([0,0,1]))  )
        
        Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
        Lkeyloop.sort()
            
        # detect loop closure and add the edge
        if params['doLoopClosure'] and Lkeyloop.index(idx)-Lkeyloop.index(previdx_loopdetect)>25:
            
            
            for gg in range(2):
                Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
                for ii in Lkeys:
                    if poseGraph.nodes[ii]['LoopDetectDone'] is False: 
                        poseGraph.nodes[ii]['LocalLoopClosed']=False
                poseGraph=pt2dproc.detectAllLoopClosures_closebyNodes(poseGraph,params,returnCopy=False,parallel=params['Loop_CLOSURE_PARALLEL'])
                poseGraph=pt2dproc.LoopCLose_CloseByNodes(poseGraph,params)
            
            
            poseGraph=pt2dproc.detectAllLoopClosures(poseGraph,params,returnCopy=False,parallel=params['Loop_CLOSURE_PARALLEL'])
            
            
            previdx_loopdetect=idx
            
            res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,Lkeyloop[0],idx,maxiter=None,algo='trf')
    
            if res.success:
                poseGraph2=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated,updateRelPoses=True)
                poseGraph = copy.deepcopy(poseGraph2)
            else:
                print("opt is failure")
                print(res)
            
            

            
                    
    else: #not a keyframe
        tpos=np.matmul(gHs,np.array([0,0,1]))

        poseGraph.add_node(idx,frametype="scan",time=idx,X=X,sHg=sHg,pos=(tpos[0],tpos[1]),color='r',LoopDetectDone=False)
        
        
        poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,H_prevframe=sHk_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Scan",color='r')
    
    
    
    
    


    
    
    
    # plotting
    # if idx%25==0 or idx==idxLast-1:
st = time.time()
pt2dplot.plot_keyscan_path(poseGraph,idx1,idx,params,makeNew=False,skipScanFrame=True,plotGraphbool=True,
                            forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)
et = time.time()
print("plotting time : ",et-st)
plt.show()
plt.pause(0.2)
plt.close("all")
        
    
    

N=len(poseGraph)
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
Lscan = list(filter(lambda x: poseGraph.nodes[x]['frametype']!="keyframe",poseGraph.nodes))
print(N,len(Lkey),len(Lscan))
df=pd.DataFrame({'type':['keyframe']*len(Lkey)+['scan']*len(Lscan),'idx':Lkey+Lscan})
df.sort_values(by=['idx'],inplace=True)
df

with open("PoseGraph-deutchesMesuemDebug-planes33.pkl",'wb') as fh:
    pkl.dump([poseGraph],fh)