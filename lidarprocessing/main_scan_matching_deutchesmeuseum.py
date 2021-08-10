# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:46:31 2020

@author: nadur
"""

#%%


#%%


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
poseData = {}

# Xr=np.zeros((len(dataset),3))
ri=0
KeyFrames=[]

params={}

params['REL_POS_THRESH']=0.5 # meters after which a keyframe is made
params['REL_ANGLE_THRESH']=15*np.pi/180
params['ERR_THRES']=3.5
params['n_components']=35
params['reg_covar']=0.002
params['BinDownSampleKeyFrame_dx']=0.05
params['BinDownSampleKeyFrame_probs']=0.15

params['doLoopClosure'] = True
params['LOOP_CLOSURE_D_THES']=31.4
params['LOOP_CLOSURE_POS_THES']=40
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
# params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
# params['LOOPCLOSE_BIN_MATCHER_L'] = 13
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = np.array([0.25,0.25],dtype=np.float64)
params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.2
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']=0.5
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']=0.4
params['LOOP_CLOSURE_COMBINE_MAX_NODES']= 16
params['offsetNodesBy'] = 2

params['NearLoopClose'] = {}
params['NearLoopClose']['PoseGrid']=None
params['NearLoopClose']['isPoseGridOffset']=True
params['NearLoopClose']['isBruteForce']=False


# meters. skip loop closure of current node if there is a loop closed node within radius along the path
params['LongLoopClose'] = {}
params['LongLoopClose']['SkipLoopCloseIfNearCLosedNodeWithin'] = 5 
params['LongLoopClose']['PoseGrid']=None
params['LongLoopClose']['isPoseGridOffset']=False
params['LongLoopClose']['isBruteForce']=True

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

idx1=0 #16000 #27103 #14340
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
        h=pt2dproc.get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
        poseGraph.add_node(idx,frametype="keyframe",clf=clf,time=idx,sHg=H,pos=(0,0),h=h,color='g',LoopDetectDone=False)
        poseData[idx]={'X':X}
        
        KeyFrame_prevIdx=idx
        KeyFrames.append(np.array([0,0,0]))
        continue
    
    # estimate pose to last keyframe
    KeyFrameClf = poseGraph.nodes[KeyFrame_prevIdx]['clf']
    Xclf = poseData[KeyFrame_prevIdx]['X']
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
        # Xd,m = pt2dproc.get0meanIcov(X)
        # weights_init = KeyFrameClf.weights_.copy()
        # means_init=KeyFrameClf.means_.copy()
        # means_init=np.matmul(sHk,np.vstack([means_init.T,np.ones(means_init.shape[0])])).T  
        # means_init=means_init[:,:2]
        # R=sHk[0:2,0:2]
        # precisions_init=KeyFrameClf.covariances_.copy()
        # for ic in range(precisions_init.shape[0]):
        #     precisions_init[ic] = nplinalg.inv(precisions_init[ic])
        #     precisions_init[ic] = nplinalg.multi_dot([R.T,precisions_init[ic],R])
        
        poseData[idx]={'X':X}
        # now delete previous scan data up-until the previous keyframe
        # this is to save space. but keep 1. Also complete pose estimation to this scan
        pt2dproc.addNewKeyFrame(poseGraph,poseData,idx,KeyFrame_prevIdx,sHg,params,keepOtherScans=False)
        poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,H_prevframe=sHk_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Key",color='k')
    
    
        KeyFrame_prevIdx = idx
        KeyFrames.append(np.matmul(gHs,np.array([0,0,1]))  )
        
        Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
        Lkeyloop.sort()
            
        # detect loop closure and add the edge
        if params['doLoopClosure'] and Lkeyloop.index(idx)-Lkeyloop.index(previdx_loopdetect)>5:
            
            
           
            poseGraph=pt2dproc.detectAllLoopClosures_closebyNodes(poseGraph,poseData,params,returnCopy=False,parallel=True)
            poseGraph=pt2dproc.LoopCLose_CloseByNodes(poseGraph,poseData,params)
            poseGraph=pt2dproc.detectAllLoopClosures(poseGraph,poseData,params,returnCopy=False,parallel=True)
            
            
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

        poseGraph.add_node(idx,frametype="scan",time=idx,sHg=sHg,pos=(tpos[0],tpos[1]),color='r',LoopDetectDone=False)
        poseData[idx]={'X':X}
        
        poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,H_prevframe=sHk_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Scan",color='r')
    
    
    
    
    


    
    
    
    # plotting
    if idx%25==0 or idx==idxLast-1:
        st = time.time()
        pt2dplot.plot_keyscan_path(poseGraph,poseData,idx1,idx,makeNew=False,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)
        et = time.time()
        print("plotting time : ",et-st)
        plt.show()
        plt.pause(0.01)
    # Lidxs = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    # plt.figure("Graph Plot")
    # axgraph.cla()
    # plotGraph(poseGraph,Lidxs,ax=axgraph)
    # plt.show()
    # plt.pause(0.1)
    
    # # plotting
    # # if idx % 1==0 or idx==len(dataset)-1:
    # plt.figure("Full Plot")adjustPoses
    # Lidxs.sort()
    # for i in Lidxs:
    #     gHs = nplinalg.inv(poseGraph.nodes[i]['sHg'])
    #     XX = poseGraph.nodes[i]['X']
    #     XX=np.matmul(gHs,np.vstack([XX.T,np.ones(XX.shape[0])])).T   
    #     ax.plot(XX[:,0],XX[:,1],'b.')
        
    # gHs=nplinalg.inv(poseGraph.nodes[idx]['sHg'])
    # # Xg=np.matmul(gHs,np.vstack([X.T,np.ones(X.shape[0])])).T   

    # # Xg=Xg[:,:2]
    # Xr[ri,:] =np.matmul(gHs,np.array([0,0,1]))   
    # ri=ri+1
    # # ax.plot(Xg[:,0],Xg[:,1],'b.')

    # ax.plot(Xr[:ri,0],Xr[:ri,1],'r')
    # ax.plot(Xr[ri-1,0],Xr[ri-1,1],'ro')
    
    # XX=np.vstack(KeyFrames)
    # ax.plot(XX[:,0],XX[:,1],'gs')
    
    # ax.set_title(str(idx))
    
    # plt.show()adjustPoses
    # plt.pause(0.1)




N=len(poseGraph)
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
Lscan = list(filter(lambda x: poseGraph.nodes[x]['frametype']!="keyframe",poseGraph.nodes))
print(N,len(Lkey),len(Lscan))
df=pd.DataFrame({'type':['keyframe']*len(Lkey)+['scan']*len(Lscan),'idx':Lkey+Lscan})
df.sort_values(by=['idx'],inplace=True)
df

with open("PoseGraph-deutchesMesuemDebug-planes33.pkl",'wb') as fh:
    pkl.dump([poseGraph,poseData],fh)

#%% Doing faster loop closure

with open("PoseGraph-deutchesMesuemDebug-planes33.pkl",'rb') as fh:
    poseGraph,poseData=pkl.load(fh)
    
params['doLoopClosure'] = True
params['LOOP_CLOSURE_D_THES']=31.4
params['LOOP_CLOSURE_POS_THES']=50
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
# params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
# params['LOOPCLOSE_BIN_MATCHER_L'] = 13
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = 0.15
params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.35
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']=0.7
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']=0.5
params['LOOP_CLOSURE_COMBINE_MAX_NODES']= 4
params['offsetNodesBy'] = 2
# params['Do_GMM_FINE_FIT']=False

# params['Do_BIN_FINE_FIT'] = False

params['Do_BIN_DEBUG_PLOT-dx']=False
params['Do_BIN_DEBUG_PLOT']= False

params['xy_hess_inv_thres']=100000000*0.4
params['th_hess_inv_thres']=100000000*0.4

params['#ThreadsLoopClose']=6

params['INTER_DISTANCE_BINS_max']=120
params['INTER_DISTANCE_BINS_dx']=1


idx1=0
nodelist = list(poseGraph.nodes)
idx1 = nodelist[0]

idx = nodelist[-1]
for nn in poseGraph.nodes:
    poseGraph.nodes[nn]['LoopDetectDone']=False

EE=[]
for ee in  poseGraph.edges:
    if poseGraph.edges[ee]['edgetype']=="Key2Key-LoopClosure":
        EE.append(ee)
        
poseGraph.remove_edges_from(EE)



pt2dplot.plot_keyscan_path(poseGraph,poseData,idx1,idx,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)


poseGraph=pt2dproc.detectAllLoopClosures(poseGraph,poseData,params,returnCopy=False)

pt2dplot.plot_keyscan_path(poseGraph,poseData,idx1,idx,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)



res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,idx1,idx,maxiter=None,algo='trf')

if res.success:
    poseGraph2=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated,updateRelPoses=True)
    # poseGraph = copy.deepcopy(poseGraph2)
else:
    print("opt is failure")
    print(res)

pt2dplot.plot_keyscan_path(poseGraph2,poseData,idx1,idx,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)
    
#%%
with open("PoseGraph-deutchesMesuemDebug-planes22-traingleLoopClose.pkl",'rb') as fh:
    poseGraph,poseData=pkl.load(fh)
    
Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    
Seqs=[]

for seq in nx.simple_cycles(poseGraph):
    seq = sorted(list(seq))
    Seqs.append(seq)
    
# Seqs=sorted(Seqs,key=lambda x: len(x),reverse=True)
Seqs=sorted(Seqs,key=lambda x: x[0])
Seqs_todo = []
for seq in Seqs:
    S=[Lkeys.index(s) for s in seq]
    if np.all(np.abs(np.diff(S))==1): 
        Seqs_todo.append(seq)
# Seqs_todo is the list of sorted decreasing sequences
# do pose optimization and update all the global poses
for i,seq in enumerate(Seqs_todo):
    # if max(seq)<=3502 or min(seq)>=3952:# def fails upto 8070 ; 3898 is bad     ; 3111 is okay  ;3952 and idx>=3502
    s1 = seq[0]
    s2 = seq[-1]
    st=time.time()
    res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,s1,s2,maxiter=None,algo='trf',xtol=1e-4)
    if res.success:
        poseGraph=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated,updateRelPoses=True)
    else:
        print("opt is failure")
        print(res)

pt2dplot.plot_keyscan_path(poseGraph,poseData,Lkeys[0],Lkeys[-1],makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)


#%%
# At this point 'sHg' is the updated combine now
# combine the points to the mid frame of the traingle
Seqs_todo=sorted(Seqs_todo,key=lambda x: len(x),reverse=True)
for i,seq in enumerate(Seqs_todo):
    if set(seq) & set(poseGraph.nodes)!=set(seq):
        continue
        
    mn = seq[int(len(seq)/2)]
    X=[poseData[mn]['X']]
    mnHg=poseGraph.nodes[mn]['sHg']
    gHmn=nplinalg.inv(mnHg)
    for nn in seq:
        if nn==mn:
            continue
        nnHg=poseGraph.nodes[nn]['sHg']
        gHnn = nplinalg.inv(nnHg)
        mnHnn = np.matmul(mnHg,gHnn)
        XX=np.matmul(mnHnn,np.vstack([poseData[nn]['X'].T,np.ones(poseData[nn]['X'].shape[0])])).T  
        X.append(XX[:,:2])
        
        
    X=pt2dproc.binnerDownSamplerProbs(X,dx=params['BinDownSampleKeyFrame_dx'],prob=0.35)
    poseData[mn]['X']=X
    res = pt2dproc.getclf(X,params,doReWtopt=True,means_init=None)
    clf=res['clf']
    poseGraph.nodes[mn]['clf']=clf
    idbmx = params['INTER_DISTANCE_BINS_max']
    idbdx=params['INTER_DISTANCE_BINS_dx']
    h=pt2dproc.get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
    poseGraph.nodes[mn]['h']=h
    
    
    ln = seq[-1]
    for sidx in poseGraph.successors(ln):
        if poseGraph.nodes[sidx]['frametype']=="keyframe" and poseGraph.edges[ln,sidx]['edgetype']=="Key2Key":
            sHln = poseGraph.edges[ln,sidx]['H']
            
            lnHg=poseGraph.nodes[ln]['sHg']
            lnHmn = np.matmul(lnHg,gHmn)
            
            sHmn = np.matmul(sHln,lnHmn)
            
            H_prevframe=poseGraph.edges[ln,sidx]['H_prevframe']
            
            serrk=poseGraph.edges[ln,sidx]['err']
            shessk_inv=poseGraph.edges[ln,sidx]['hess_inv']
            poseGraph.add_edge(mn,sidx,H=sHmn,H_prevframe=H_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Key",color='k')

            break
    
    fn = seq[0]
    for pidx in poseGraph.predecessors(fn):
        if poseGraph.nodes[pidx]['frametype']=="keyframe" and poseGraph.edges[pidx,fn]['edgetype']=="Key2Key":
            fnHp = poseGraph.edges[pidx,fn]['H']
            
            fnHg=poseGraph.nodes[fn]['sHg']
            gHfn=nplinalg.inv(fnHg)
            mnHfn = np.matmul(mnHg,gHfn)
            
            mnHp = np.matmul(mnHfn,fnHp)
            
            H_prevframe=poseGraph.edges[pidx,fn]['H_prevframe']
            
            serrk=poseGraph.edges[pidx,fn]['err']
            shessk_inv=poseGraph.edges[pidx,fn]['hess_inv']
            poseGraph.add_edge(pidx,mn,H=mnHp,H_prevframe=H_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Key",color='k')

            break
    
    for nn in seq:
        if nn!=mn:
            poseGraph.remove_node(nn)
            poseData.pop(nn,None)
                
Lkeyloop = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))

for ee in Lkeyloop:
    poseGraph.remove_edge(ee[0],ee[1])
    
Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes)) 
pt2dplot.plot_keyscan_path(poseGraph,poseData,Lkeys[0],Lkeys[-1],makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)


    
with open("PoseGraph-deutchesMesuemDebug-planes22-traingleLoopClose-combined.pkl",'wb') as fh:
    pkl.dump([poseGraph,poseData],fh)
    
#%% Now do global loop closure
with open("PoseGraph-deutchesMesuemDebug-planes22-traingleLoopClose-combined.pkl",'rb') as fh:
    poseGraph,poseData=pkl.load(fh)
    

params['LOOP_CLOSURE_D_THES']=30.3
params['LOOP_CLOSURE_POS_THES']=20
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.5
params['LOOPCLOSE_BIN_MAXOVRL_FRAC']=0.50

params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = 0.15

params['LOOP_CLOSURE_COMBINE_MAX_NODES']= 5
params['LOOP_CLOSURE_COMBINE_TRIANGLES']= True

idx1=0


Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
Lkeys.sort()
for idx in Lkeys:
    poseGraph.nodes[idx]['LoopDetectDone'] = False
    
    # idbmx = params['INTER_DISTANCE_BINS_max']
    # idbdx=params['INTER_DISTANCE_BINS_dx']
    # X = poseData[idx]['X']
    # h=pt2dproc.get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
    # poseGraph.nodes[idx]['h']=h
    
    
for idx in Lkeys:
    if poseGraph.nodes[idx]['LoopDetectDone'] is False:          
        h1=poseGraph.nodes[idx]['h']
        p1=poseGraph.nodes[idx]['pos']
        # previdx < Lkeys[Lkeys.index(idx)-1] and 
        LPkeys = Lkeys[:max([Lkeys.index(idx)-2,0])]
        for previdx in LPkeys:
            # if previdx>=idx:
            #     continue
            
            if poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:
                
                h2=poseGraph.nodes[previdx]['h']
                
                p2=poseGraph.nodes[previdx]['pos']
                d=nplinalg.norm(h1-h2,ord=1)
                c1 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=params['LOOP_CLOSURE_POS_THES']
                c2 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)>=params['LOOP_CLOSURE_POS_MIN_THES']
                if d<=params['LOOP_CLOSURE_D_THES'] and c1 and c2:
                    # add loop closure edge
                    # print("Potential Loop closure")
                    st=time.time()
                    piHi,pi_err_i,mbin,mbinfrac,hess_inv_err_i,mbinfrac_ActiveOvrlp=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,poseData,idx,previdx,params)
                    print(idx,previdx,time.time()-st)
                    # piHi,pi_err_i,hess_inv_err_i=0,0,0
                    posematch={'mbin':mbin,'mbinfrac':mbinfrac,'mbinfrac_ActiveOvrlp':mbinfrac_ActiveOvrlp}
                    a1=pi_err_i < params['LOOP_CLOSURE_ERR_THES']
                    a2=mbinfrac>=params['LOOPCLOSE_BIN_MIN_FRAC']
                    a3=mbinfrac_ActiveOvrlp>=params['LOOPCLOSE_BIN_MAXOVRL_FRAC']
                    if a3:
                        poseGraph.add_edge(idx,previdx,H=piHi,err = pi_err_i,posematch=posematch,hess_inv = hess_inv_err_i,edgetype="Key2Key-LoopClosure",d=d,color='b')
                        print("Added")
                        # res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,Lkeyloop[max([0,len(Lkeyloop)-1000])],idx,maxiter=None,algo='trf')
                        
        
        poseGraph.nodes[idx]['LoopDetectDone'] = True

pt2dplot.plot_keyscan_path(poseGraph,poseData,Lkeys[0],Lkeys[-1],makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

with open("PoseGraph-deutchesMesuemDebug-planes22-traingleLoopClose-combined-loopclose.pkl",'wb') as fh:
    pkl.dump([poseGraph,poseData],fh)
#%% Optimize poses
with open("PoseGraph-deutchesMesuemDebug-planes22-traingleLoopClose-combined-loopclose.pkl",'rb') as fh:
    poseGraph,poseData=pkl.load(fh)


Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,Lkeys[0],Lkeys[-1],maxiter=None,algo='lm')
if res.success:
    poseGraph=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated,updateRelPoses=True)
else:
    print("opt is failure")
    print(res)
    

pt2dplot.plot_keyscan_path(poseGraph,poseData,Lkeys[0],Lkeys[-1],makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

#%%
Lkeyloop = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))

for idx, previdx in Lkeyloop:
    # st=time.time()
    # piHi,pi_err_i,mbin,mbinfrac,hess_inv_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,poseData,idx,previdx,params)
    # et=time.time()
    # print("Mathc time = ",et-st)
    mbinfrac=poseGraph.edges[idx,previdx]['posematch']['mbinfrac']
    mbinfrac_ActiveOvrlp=poseGraph.edges[idx,previdx]['posematch']['mbinfrac_ActiveOvrlp']
    
    piHi=poseGraph.edges[idx,previdx]['H']
    
    pt2dplot.plotcomparisons(poseGraph,poseData,idx,previdx,H12=nplinalg.inv(piHi),err=mbinfrac_ActiveOvrlp) #nplinalg.inv(piHi) 
    fig = plt.figure("ComparisonPlot")
    fig.savefig("loopdetect-%d-%d.png"%(idx, previdx))
    plt.close(fig)


    
#%%
plt.close("all")
with open("PoseGraph-deutchesMesuemDebug-planes2.pkl",'rb') as fh:
    poseGraph,poseData=pkl.load(fh)

Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
idx1 = Lkeyloop[0]
idx=Lkeyloop[-1]

poseGraph.edges[6623,2566]
poseGraph.edges[6623,2535]



# res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,idx1,idx,algo='trf')
res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,Lkeyloop[max([0,len(Lkeyloop)-1000])],idx,maxiter=None,algo='trf')

if res.success:
    pass
else:
    print("opt is failure")
    print(res)
poseGraph2=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated)
# poseGraph = copy.deepcopy(poseGraph2)

pt2dplot.plot_keyscan_path(poseGraph,poseData,idx1,idx,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

plt.show()

pt2dplot.plot_keyscan_path(poseGraph2,poseData,idx1,idx,makeNew=True,skipScanFrame=True,plotGraphbool=False,
                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)
    
#%%
G = poseGraph.subgraph(Lkeyloop[:12])
for seq in nx.simple_cycles(G):
    seq = sorted(list(seq))
    print(seq)
# nx.find_cycle(poseGraph,source=Lkeyloop[1])

#%%
# plt.close("all")
# previdx= 20624
# idx = 20757

# previdx= 20448
# idx = 20716

# previdx= 20435
# idx = 20716

# previdx= 20370
# idx = 20701

# previdx= 16000
# idx = 16044
previdx= 2535
idx = 6623


# Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
# Lkey = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))

params={}

params['REL_POS_THRESH']=0.5 # meters after which a keyframe is made
params['ERR_THRES']=2.5
params['n_components']=35
params['reg_covar']=0.002



params['doLoopClosure'] = True
params['LOOP_CLOSURE_D_THES']=0.3
params['LOOP_CLOSURE_POS_THES']=25
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
params['LOOPCLOSE_BIN_MATCHER_L'] = 13
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = 0.25
params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.5

params['Do_GMM_FINE_FIT']=True

params['Do_BIN_FINE_FIT'] = False

params['Do_BIN_DEBUG_PLOT-dx']=False
params['Do_BIN_DEBUG_PLOT']= False

params['xy_hess_inv_thres']=100000000*0.4
params['th_hess_inv_thres']=100000000*0.4
params['#ThreadsLoopClose']=6

params['INTER_DISTANCE_BINS_max']=120
params['INTER_DISTANCE_BINS_dx']=1
st=time.time()
piHi,pi_err_i,mbin,mbinfrac,hess_inv_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,poseData,idx,previdx,params)
et=time.time()
print("Mathc time = ",et-st)

pt2dplot.plotcomparisons(poseGraph,poseData,idx,previdx,H12=nplinalg.inv(piHi),err=mbinfrac) #nplinalg.inv(piHi) 
    

#%% 
Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))

for idx in Lkeys:
    if idx>=23140 and idx<=24525:
        pass
    else:
        continue
    for previdx in Lkeys:
        if previdx >=15760 and previdx <= 19100:
            pass
        else:
            continue
        
        st=time.time()
        # piHi,pi_err_i,mbin,mbinfrac,hess_inv_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,poseData,idx,previdx,params)
        posematch=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,poseData,idx,previdx,params)
        piHi = posematch['H']
        mbinfrac = posematch['mbinfrac']
        et=time.time()
        print("Mathc time = ",et-st)
        
        pt2dplot.plotcomparisons(poseGraph,poseData,idx,previdx,H12=nplinalg.inv(piHi),err=mbinfrac) #nplinalg.inv(piHi) 
        fig = plt.figure("ComparisonPlot")
        fig.savefig("loopdetec-%d-%d.png"%(idx, previdx))
        plt.close(fig)
#%%
piHi,pi_err_i,mbin,mbinfrac,hess_inv_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,poseData,757,670,params)