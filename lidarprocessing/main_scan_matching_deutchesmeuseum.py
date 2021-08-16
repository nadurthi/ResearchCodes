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
    # if idx >= 20079 and idx<=20089:
    #     return None
    try:
        with open(os.path.join(scanfilefolder,'scan_%d.pkl'%idx),'rb') as F:
            scan=pkl.load(F)
    except:
        return None
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
params['ERR_THRES']=2.5
params['n_components']=35
params['reg_covar']=0.002

params['BinDownSampleKeyFrame_dx']=0.05
params['BinDownSampleKeyFrame_probs']=0.1

params['Plot_BinDownSampleKeyFrame_dx']=0.05
params['Plot_BinDownSampleKeyFrame_probs']=0.001

params['doLoopClosure'] = True
params['doLoopClosureLong'] = True

params['Loop_CLOSURE_PARALLEL'] = True
params['LOOP_CLOSURE_D_THES']=31.4
params['LOOP_CLOSURE_POS_THES']=25
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
# params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
# params['LOOPCLOSE_BIN_MATCHER_L'] = 13
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = np.array([0.25,0.25],dtype=np.float64)
params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.2
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']=0.6
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']=0.5
params['LOOP_CLOSURE_COMBINE_MAX_NODES']= 16
params['offsetNodesBy'] = 2
params['MAX_NODES_ADJ_COMBINE']=5

params['NearLoopClose'] = {}
params['NearLoopClose']['Method']='GMM'
params['NearLoopClose']['PoseGrid']=None #pt2dproc.getgridvec(np.linspace(-np.pi/12,np.pi/12,3),np.linspace(-1,1,3),np.linspace(-1,1,3))
params['NearLoopClose']['isPoseGridOffset']=True
params['NearLoopClose']['isBruteForce']=False


# meters. skip loop closure of current node if there is a loop closed node within radius along the path
params['LongLoopClose'] = {}
params['LongLoopClose']['Method'] = 'GMM'
params['LongLoopClose']['SkipLoopCloseIfNearCLosedNodeWithin'] = 5 
params['LongLoopClose']['PoseGrid']= None
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

idx1=0 #19970 #16000 #27103 #14340
idxLast = Nframes
previdx_loopclosure = idx1
previdx_loopdetect = idx1
previdx_loopdetect_long=idx1
for idx in range(idx1,idxLast): 
    # ax.cla()
    if idx>=20083 and idx <=20085:
        continue
    X=getscanpts(idx)
    if X is None:
        continue
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
    
    
    if (idx-KeyFrame_prevIdx)<=1:
        sHk_prevframe = np.identity(3)
    elif KeyFrame_prevIdx==previdx:
        sHk_prevframe = np.identity(3)
    else:
        sHk_prevframe = poseGraph.edges[KeyFrame_prevIdx,previdx]['H']
        
    # sHk_prevframe = np.identity(3)
    
        
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
        
        # pdb.set_trace()
        # bin match key frame to keyframe
        dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
        Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(Xclf,X,dxcomp)
        activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
        H12=poseGraph.edges[KeyFrame_prevIdx,idx]['H']
        posematch=pt2dproc.eval_posematch(H12,X,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
        poseGraph.edges[KeyFrame_prevIdx,idx]['posematch']=posematch
        print("posematch['mbinfrac_ActiveOvrlp']=",posematch['mbinfrac_ActiveOvrlp'])
        if posematch['mbinfrac_ActiveOvrlp']<0.5:
            posematch = pt2dproc.poseGraph_keyFrame_matcher_binmatch(poseGraph,KeyFrame_prevIdx,idx,params,DoCLFmatch=True,dx0=0.8,L0=1,th0=np.pi/12,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
            poseGraph.edges[KeyFrame_prevIdx,idx]['H']=posematch['H']
            poseGraph.edges[KeyFrame_prevIdx,idx]['posematch']=posematch
            
            sHg = np.matmul(posematch['H'],kHg)
            gHs=nplinalg.inv(sHg)    
            tpos=np.matmul(gHs,np.array([0,0,1])) 
            poseGraph.nodes[idx]['pos']=(tpos[0],tpos[1])
            poseGraph.nodes[idx]['sHg']=sHg
        
        # pt2dplot.plotcomparisons(poseGraph,KeyFrame_prevIdx,idx,UseLC=False,H12=nplinalg.inv( sHk) ,err=serrk)
        # fig = plt.figure("ComparisonPlot")
        # fig.savefig("ComparisonPlot-%d-%d.png"%(KeyFrame_prevIdx, idx))
        # plt.close(fig)
        
        KeyFrame_prevIdx = idx
        KeyFrames.append(np.matmul(gHs,np.array([0,0,1]))  )
        
        Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
        Lkeyloop.sort()
            
        # detect loop closure and add the edge
        if params['doLoopClosure'] and Lkeyloop.index(idx)-Lkeyloop.index(previdx_loopdetect)>20:
            
            poseGraph=pt2dproc.detectAllLoopClosures_closebyNodes(poseGraph,params,returnCopy=False,parallel=params['Loop_CLOSURE_PARALLEL'])
            poseGraph=pt2dproc.LoopCLose_CloseByNodes(poseGraph,params)
            
            # # bin match key frame to keyframe
            # Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
            # for KeyFrameIdx11 in Lkeyloop:
            #     for idx11 in poseGraph.successors(KeyFrameIdx11):
            #         if poseGraph.edges[KeyFrameIdx11,idx11].get('DoneBinMatch',False) is False:
            #             posematch = pt2dproc.poseGraph_keyFrame_matcher_binmatch(poseGraph,KeyFrameIdx11,idx11,params,DoCLFmatch=True,dx0=1.5,L0=10,th0=np.pi/3,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
            #             poseGraph.edges[KeyFrameIdx11,idx11]['H']=posematch['H']
            #             poseGraph.edges[KeyFrameIdx11,idx11]['DoneBinMatch']=True
                        
            previdx_loopdetect=idx
        
            # Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
            # Lkeyloop.sort()
       
        
            # mm=10000
            # cc=idx
            # for m in Lkeyloop:  
            #     if abs(m-previdx_loopdetect_long)<mm:
            #         cc=m
            #         mm=abs(m-previdx_loopdetect_long)
                
            # if params['doLoopClosureLong'] and Lkeyloop.index(idx)-Lkeyloop.index(cc)>10:
    
            poseGraph=pt2dproc.detectAllLoopClosures(poseGraph,params,returnCopy=False,parallel=params['Loop_CLOSURE_PARALLEL'])
            
            
            
            
            res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,Lkeyloop[0],idx,maxiter=None,algo='trf')
    
            if res.success:
                poseGraph2=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated,updateRelPoses=True)
                poseGraph = copy.deepcopy(poseGraph2)
            else:
                print("opt is failure")
                print(res)
            
            previdx_loopdetect_long=idx    
                
    else: #not a keyframe
        tpos=np.matmul(gHs,np.array([0,0,1]))

        poseGraph.add_node(idx,frametype="scan",time=idx,X=X,sHg=sHg,pos=(tpos[0],tpos[1]),color='r',LoopDetectDone=False)
        
        
        poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,H_prevframe=sHk_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Scan",color='r')
    
    
    
        # pt2dplot.plotcomparisons(poseGraph,KeyFrame_prevIdx,idx,UseLC=False,H12=nplinalg.inv( sHk) ,err=serrk)
        # fig = plt.figure("ComparisonPlot")
        # fig.savefig("ComparisonPlot-%d-%d.png"%(KeyFrame_prevIdx, idx))
        # plt.close(fig)
    


    previdx = idx
    
    
    # plotting
    if idx%25==0 or idx==idxLast-1:
        st = time.time()
        pt2dplot.plot_keyscan_path(poseGraph,idx1,idx,params,makeNew=False,skipScanFrame=True,plotGraphbool=True,
                                    forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)
        et = time.time()
        print("plotting time : ",et-st)
        plt.show()
        plt.pause(0.01)
        
    
    

N=len(poseGraph)
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
Lscan = list(filter(lambda x: poseGraph.nodes[x]['frametype']!="keyframe",poseGraph.nodes))
print(N,len(Lkey),len(Lscan))
df=pd.DataFrame({'type':['keyframe']*len(Lkey)+['scan']*len(Lscan),'idx':Lkey+Lscan})
df.sort_values(by=['idx'],inplace=True)
df

with open("PoseGraph-deutchesMesuemDebug-planes33.pkl",'wb') as fh:
    pkl.dump([poseGraph,params],fh)

#%% scan to scan test
Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
Lkeyloop.sort()
        
KeyFrameIdx = 20081
idx = 20084

H21_est = poseGraph.edges[KeyFrameIdx,idx]['H']

params['n_components']=35
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = np.array([0.25,0.25],dtype=np.float64)

Xclf=getscanpts(KeyFrameIdx)
Xidx=getscanpts(idx)

res = pt2dproc.getclf(Xclf,params,doReWtopt=True,means_init=None)
KeyFrameClf=res['clf']

poseGraph.edges[KeyFrameIdx,idx]['H']

st=time.time()
posematch = pt2dproc.poseGraph_keyFrame_matcher_binmatch(poseGraph,KeyFrameIdx,idx,params,dx0=1,L0=3,th0=np.pi/6,DoCLFmatch=True,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
et=time.time()
print("time taken matching = ",et-st)
# pt2dplot.plotcomparisons_points(Xclf,Xidx,KeyFrameIdx,idx,KeyFrameClf,UseLC=False,H21_est=H21_est,H12=nplinalg.inv( posematch['H']) ,err=posematch['mbinfrac_ActiveOvrlp'])

sHk=np.identity(3)
sHk[0,2]=1
# sHk[1,2]=1
# sHk_corrected,serrk,shessk_inv = pt2dproc.scan2keyframe_match(KeyFrameClf,Xclf,Xidx,params,sHk=sHk)   
# pt2dplot.plotcomparisons_points(Xclf,Xidx,KeyFrameIdx,idx,KeyFrameClf,UseLC=False,H12=nplinalg.inv( sHk_corrected) ,err=serrk)
# fig = plt.figure("ComparisonPlot")
# fig.savefig("ComparisonPlot-%d-%d.png"%(KeyFrame_prevIdx, idx))
# plt.close(fig)
    


#%% Doing faster loop closure

with open("PoseGraph-deutchesMesuemDebug-planes33.pkl",'rb') as fh:
    poseGraph,params=pkl.load(fh)
    



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



pt2dplot.plot_keyscan_path(poseGraph,idx1,idx,params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)


poseGraph=pt2dproc.detectAllLoopClosures(poseGraph,params,returnCopy=False)

pt2dplot.plot_keyscan_path(poseGraph,idx1,idx,params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)



res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,idx1,idx,maxiter=None,algo='trf')

if res.success:
    poseGraph2=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated,updateRelPoses=True)
    # poseGraph = copy.deepcopy(poseGraph2)
else:
    print("opt is failure")
    print(res)

pt2dplot.plot_keyscan_path(poseGraph2,idx1,idx,params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)
    


#%%
Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
27905
27844
27905
27843
27820
Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
idx=27905
idx_p1=Lkeyloop[Lkeyloop.index(idx)+1]
idx_m1=Lkeyloop[Lkeyloop.index(idx)-1]


H21_est = poseGraph.edges[idx,idx_p1]['H']
pt2dplot.plotcomparisons(poseGraph,idx,idx_p1,UseLC=False,H12=nplinalg.inv(H21_est),err=0) #nplinalg.inv(piHi) 


for idx, previdx in Lkeyloop_edges:
    if idx>=27333 and idx<=28751:
        pass
    else:
        continue
    
    # st=time.time()
    # piHi,pi_err_i,mbin,mbinfrac,hess_inv_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,idx,previdx,params)
    # et=time.time()
    # print("Mathc time = ",et-st)
    # posematch=pt2dproc.poseGraph_keyFrame_matcher_long(poseGraph,idx,previdx,params,params['LongLoopClose']['PoseGrid'],
    #                                                      params['LongLoopClose']['isPoseGridOffset'],
    #                                                      params['LongLoopClose']['isBruteForce'])
    
    posematch=poseGraph.edges[idx,previdx]['posematch']
    mbinfrac=posematch['mbinfrac']
    mbinfrac_ActiveOvrlp=posematch['mbinfrac_ActiveOvrlp']
    
    piHi=posematch['H']
    
    pt2dplot.plotcomparisons(poseGraph,idx,previdx,UseLC=True,H12=nplinalg.inv(piHi),err=mbinfrac_ActiveOvrlp) #nplinalg.inv(piHi) 
    fig = plt.figure("ComparisonPlot")
    fig.savefig("loopdetect-%d-%d.png"%(idx, previdx))
    plt.close(fig)

#%%
with open("PoseGraph-deutchesMesuemDebug-planes33.pkl",'rb') as fh:
    poseGraph,=pkl.load(fh)
    
idx=27827
previdx=27709
params['LongLoopClose']['PoseGrid']= None #pt2dproc.getgridvec(np.linspace(-np.pi/6,np.pi/6,10),np.linspace(-5,5,3),np.linspace(-5,5,3))
params['LongLoopClose']['isPoseGridOffset']=True
params['LongLoopClose']['isBruteForce']=False

X1,clf1=pt2dproc.getCombinedNode(poseGraph,idx,5,params,Doclf=True)
poseGraph.nodes[idx]['clflc']=clf1
poseGraph.nodes[idx]['Xlc']=X1
poseGraph.nodes[idx]['DoneAdjCombine']=True

X1,clf1=pt2dproc.getCombinedNode(poseGraph,previdx,5,params,Doclf=True)
poseGraph.nodes[previdx]['clflc']=clf1
poseGraph.nodes[previdx]['Xlc']=X1
poseGraph.nodes[previdx]['DoneAdjCombine']=True

posematch=pt2dproc.poseGraph_keyFrame_matcher_long(poseGraph,idx,previdx,params,params['LongLoopClose']['PoseGrid'],
                                                          params['LongLoopClose']['isPoseGridOffset'],
                                                          params['LongLoopClose']['isBruteForce'])
    
# posematch=poseGraph.edges[idx,previdx]['posematch']
mbinfrac=posematch['mbinfrac']
mbinfrac_ActiveOvrlp=posematch['mbinfrac_ActiveOvrlp']

piHi=posematch['H']

pt2dplot.plotcomparisons(poseGraph,idx,previdx,UseLC=True,H12=nplinalg.inv(piHi),err=mbinfrac_ActiveOvrlp) #nplinalg.inv(piHi)
    
#%%
plt.close("all")
with open("PoseGraph-deutchesMesuemDebug-planes2.pkl",'rb') as fh:
    poseGraph=pkl.load(fh)

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

pt2dplot.plot_keyscan_path(poseGraph,idx1,idx,params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

plt.show()

pt2dplot.plot_keyscan_path(poseGraph2,idx1,idx,params,makeNew=True,skipScanFrame=True,plotGraphbool=False,
                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)
    


#%% 
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))

plt.close("all")
# pt2dplot.plot_keyscan_path(poseGraph,Lkey[0],Lkey[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
#                                     forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

def plotGmmX(ax,clf,X):
    if ax is None:
        fig=plt.figure()
        ax = fig.subplots(nrows=1, ncols=1)
    if X is not None:
        ax.plot(X[:,0],X[:,1],'k.')
    for i in range(clf.n_components):
        # print("ok")
        m = clf.means_[i]
        P = clf.covariances_[i]
        Xe= utpltgmshp.getCovEllipsePoints2D(m,P,nsig=1,N=100)
        ax.plot(Xe[:,0],Xe[:,1],'g')
 
        


def getCombinedNode2(poseGraph,idx,nn):
    G=poseGraph.subgraph(Lkey[Lkey.index(idx)-nn:Lkey.index(idx)+nn])
    for lp in list(filter(lambda x: idx in x,list(nx.simple_cycles(G)))):
        MU=[poseGraph.nodes[idx]['clf'].means_]
        P=[poseGraph.nodes[idx]['clf'].covariances_]
        W=[poseGraph.nodes[idx]['clf'].weights_]
        X=[poseGraph.nodes[idx]['X']]
        
        
        for jj in lp:
            if jj!=idx:
                if (jj,idx) in poseGraph.edges:
                    iHj = poseGraph.edges[jj,idx]['H']
                else:
                    iHj = nplinalg.inv(poseGraph.edges[idx,jj]['H'])
                
                Xj=poseGraph.nodes[jj]['X']
                res = pt2dproc.getclf(Xj,params,doReWtopt=True)
                clfj=res['clf']
                # clfj = poseGraph.nodes[jj]['clf']
                
                # plotGmmX(None,clfj,Xj)
                
                
                XX=np.matmul(iHj,np.vstack([Xj.T,np.ones(Xj.shape[0])])).T 
                X.append(XX[:,:2])
                
                # Mj = clfj.means_
                # Pj = clfj.covariances_
                # Wj = clfj.weights_
                
                # MM=np.matmul(iHj,np.vstack([Mj.T,np.ones(Mj.shape[0])])).T 
                # MM=MM[:,:2]    
        # fig=plt.figure()
        # ax = fig.subplots(nrows=1, ncols=1)
        # plotGmmX(ax,clf,X)
                # MU.append(MM)
                # R=iHj[0:2,0:2]
                # for ic in range(Pj.shape[0]):
                #     Pj[ic] = nplinalg.multi_dot([R,Pj[ic],R.T])
                # P.append(Pj)
                # W.append(Wj)
        
        # MU=np.vstack(MU)
        # P=np.concatenate(P,axis=0)
        # W=np.hstack(W)
        X=np.vstack(X)
        
        X=np.ascontiguousarray(X,dtype=dtype)
        # MU=np.ascontiguousarray(MU,dtype=dtype)
        # P=np.ascontiguousarray(P,dtype=dtype)
        # W=np.ascontiguousarray(W,dtype=dtype)
        
        res = pt2dproc.getclf(X,params,doReWtopt=True)
        clf = res['clf']
        
        # res=pt2dproc.optWts1(X,MU,P,W)
        # W=res.x
        # clf=pt2dproc.Clf(MU,P,W)
        
        
        
        # fig=plt.figure()
        # ax = fig.subplots(nrows=1, ncols=1)
        # plotGmmX(ax,clf,X)
            
        # plt.show()    
        # break         

#%%
jj=18278
Xj=poseGraph.nodes[jj]['X']
res = pt2dproc.getclf(Xj,params,doReWtopt=True)
clfj=res['clf']
# clfj = poseGraph.nodes[jj]['clf']
plotGmmX(None,clfj,Xj)
#%% 
with open("PoseGraph-deutchesMesuemDebug-planes33.pkl",'rb') as fh:
    poseGraph,=pkl.load(fh)

    
Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))

for idx in Lkeys:
    if idx>=27333 and idx<=28751:
        pass
    else:
        continue
    for previdx in Lkeys:
    #     if previdx >=18141 and previdx <= 18770:
    #         pass
    #     else:
    #         continue
        if (idx,previdx) in poseGraph.edges:
            st=time.time()
            posematch=poseGraph.edges[idx,previdx]['posematch']
            # posematch=pt2dproc.poseGraph_keyFrame_matcher_long(poseGraph,idx,previdx,params,params['LongLoopClose']['PoseGrid'],
            #                                                  params['LongLoopClose']['isPoseGridOffset'],
            #                                                  params['LongLoopClose']['isBruteForce'])
            piHi = posematch['H']
            mbinfrac = posematch['mbinfrac']
            et=time.time()
            print("Mathc time = ",et-st)
            
            pt2dplot.plotcomparisons(poseGraph,idx,previdx,H12=nplinalg.inv(piHi),err=mbinfrac) #nplinalg.inv(piHi) 
            fig = plt.figure("ComparisonPlot")
            fig.savefig("loopdetec-%d-%d.png"%(idx, previdx))
            plt.close(fig)
#%%
