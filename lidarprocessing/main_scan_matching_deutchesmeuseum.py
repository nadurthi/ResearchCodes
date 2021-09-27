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

# scanfilepath = 'lidarprocessing/datasets/DeutchMeuseum/b2-2016-04-27-12-31-41.pkl'
# scanfilefolder = 'lidarprocessing/datasets/DeutchMeuseum/b2-2016-04-27-12-31-41/'

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


class IntelData:
    def __init__(self):
        intelfile = "lidarprocessing/datasets/Freiburg/Intel Research Lab.clf"
        ff=open(intelfile)
        self.inteldata=ff.readlines()
        self.inteldata = list(filter(lambda x: '#' not in x,self.inteldata))
        self.flaserdata = list(filter(lambda x: 'FLASER' in x[:8],self.inteldata))
        self.odomdata = list(filter(lambda x: 'ODOM' in x[:8],self.inteldata))

        
    def __len__(self):
        return len(self.flaserdata)
    
    def getflaser(self,idx):
        g = self.flaserdata[idx]
        glist = g.strip().split(' ')
        # print(glist)
        cnt = int(glist[1])
        rngs =[]
        ths=np.arange(0,1*np.pi,1*np.pi/180)
        for i in range(cnt):
            rngs.append(float(glist[2+i]))
        odom=np.array([float(ss) for ss in glist[182:188]])
        rngs = np.array(rngs)
        p=np.vstack([np.cos(ths),np.sin(ths)])
        ptset = rngs.reshape(-1,1)*p.T
        rngidx = (rngs> (0+0.1) ) & (rngs< (25))
        return ptset[rngidx,:],odom
    
dataset = IntelData()       
def getscanpts_intel(idx):
    # ranges = dataset[i]['ranges']
    
    X,odom=dataset.getflaser(idx)
    
    # now filter silly points
    # tree = KDTree(X, leaf_size=5)
    # cnt = tree.query_radius(X, 0.0125,count_only=True) 
    # X = X[cnt>=2,:]
    
    # cnt = tree.query_radius(X, 0.015,count_only=True) 
    # X = X[cnt>=5,:]
    
    return X,odom

getscanpts = getscanpts_deutches

# plt.figure()
# Xpos=[]
# for i in range(len(dataset)):
#     Xk,odom = getscanpts(i)
#     # H=nbpt2Dproc.getHmat(odom[2],odom[0:2])
#     # Xk=np.matmul(H,np.vstack([X.T,np.ones(X.shape[0])])).T  
#     # Xk=Xk[:,:2]
#     # Xpos.append(odom[0:2])
    
#     # Xpos_arr = np.array(Xpos)
#     plt.plot(Xk[:,0],Xk[:,1],'b.')
#     # plt.plot(Xpos_arr[:,0],Xpos_arr[:,1],'r')
#     plt.title(str(i))
#     plt.pause(0.2)
#     plt.cla()
    
#%% TEST::::::: Pose estimation by keyframe
plt.close("all")
poses=[]
poseGraph = nx.DiGraph()


# Xr=np.zeros((len(dataset),3))
ri=0
KeyFrames=[]

params={}

params['REL_POS_THRESH']=49# meters after which a keyframe is made
params['REL_ANGLE_THRESH']=145*np.pi/180
params['ERR_THRES']=15
params['n_components']=35
params['reg_covar']=0.002

params["Key2Key_Overlap"]=0.3
params["Scan2Key_Overlap"]=0.3

params['Key2KeyBinMatch_dx0']=2
params['Key2KeyBinMatch_L0']=7
params['Key2KeyBinMatch_th0']=np.pi/4

params['BinDownSampleKeyFrame_dx']=0.15
params['BinDownSampleKeyFrame_probs']=0.05

params['Plot_BinDownSampleKeyFrame_dx']=0.15
params['Plot_BinDownSampleKeyFrame_probs']=0.0001

params['doLoopClosure'] = False
params['doLoopClosureLong'] = False

params['Loop_CLOSURE_PARALLEL'] = True
params['LOOP_CLOSURE_D_THES']=31.4
params['LOOP_CLOSURE_POS_THES']=30
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
# params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
# params['LOOPCLOSE_BIN_MATCHER_L'] = 13
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = np.array([0.15,0.15],dtype=np.float64)

params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.2
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']=0.6
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']=0.4
params['LOOP_CLOSURE_COMBINE_MAX_NODES']= 8

params['offsetNodesBy'] = 0


params['MAX_NODES_ADJ_COMBINE']=5
params["USE_Side_Combine"]=True
params["Side_Combine_Overlap"]=0.3


params['NearLoopClose'] = {}
params['NearLoopClose']['Method']='GMM'
params['NearLoopClose']['PoseGrid']=None #pt2dproc.getgridvec(np.linspace(-np.pi/12,np.pi/12,3),np.linspace(-1,1,3),np.linspace(-1,1,3))
params['NearLoopClose']['isPoseGridOffset']=True
params['NearLoopClose']['isBruteForce']=False


# meters. skip loop closure of current node if there is a loop closed node within radius along the path
params['LongLoopClose'] = {}
params['LongLoopClose']['Method'] = 'GMM'
params['LongLoopClose']['SkipLoopCloseIfNearCLosedNodeWithin'] = 5 
A=pt2dproc.getgridvec([0],np.linspace(-5,5,5),np.linspace(-5,5,5))
ind = np.lexsort((np.abs(A[:,0]),np.abs(A[:,1]),np.abs(A[:,2])))
params['LongLoopClose']['PoseGrid']= None #A[ind]
params['LongLoopClose']['isPoseGridOffset']=True
params['LongLoopClose']['isBruteForce']=False
params['LongLoopClose']['Bin_Match_dx0'] = 2
params['LongLoopClose']['Bin_Match_L0'] = 7
params['LongLoopClose']['Bin_Match_th0'] = np.pi/4
params['LongLoopClose']['DoCLFmatch'] = True

params['LongLoopClose']['AlongPathNearFracCountNodes'] = 0.3
params['LongLoopClose']['AlongPathNearFracLength'] = 0.3
params['LongLoopClose']['#TotalRandomPicks'] = 10
params['LongLoopClose']['AdjSkipList'] = 3
params['LongLoopClose']['TotalCntComp'] = 100

# params['Do_GMM_FINE_FIT']=False

# params['Do_BIN_FINE_FIT'] = False

params['Do_BIN_DEBUG_PLOT-dx']=False
params['Do_BIN_DEBUG_PLOT']= False

params['xy_hess_inv_thres']=100000000*0.4
params['th_hess_inv_thres']=100000000*0.4


params['#ThreadsLoopClose']=8

params['INTER_DISTANCE_BINS_max']=120
params['INTER_DISTANCE_BINS_dx']=1


params['LOOPCLOSE_AFTER_#KEYFRAMES']=2



timeMetrics={'scan2keyMain':[],'addNewKeyScan':[],'SendPlot':[],'SendPoseGraphLoop':[],'RecievePoseGraphLoop':[],
                          'UpdatePoseGraphLoop':[],'PrevPrevScanPtsStack':[],'PrevScanPtsStack':[],'PrevPrevScanPtsCombine':[],'PrevScanPtsCombine':[],'NewKeyFrameClf':[],'scan2keyNew':[],'OverlapScan2keyNew':[]}
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
    # if idx>=20083 and idx <=20085:
    #     continue
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
    
    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(Xclf,X,dxcomp)
    activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    posematch=pt2dproc.eval_posematch(sHk,X,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
    posematch['method']='GMMmatch'
    posematch['when']="Scan to key in main"
    
    print("idx = ",idx," Error = ",serrk," , and time taken = ",et-st," posematch=",posematch['mbinfrac_ActiveOvrlp'])
    
    
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
    if serrk>params['ERR_THRES'] or nplinalg.norm(sHk[:2,2])>params['REL_POS_THRESH'] or posematch['mbinfrac_ActiveOvrlp']<params["Scan2Key_Overlap"] or thdiff>params['REL_ANGLE_THRESH']:
        print("New Keyframe")
        st = time.time()
        # pt2dproc.addNewKeyFrameAndScan(self.poseGraph,KeyFrameClf,Xclf,XprevScan,idxprevScan,self.KeyFrame_prevIdx,sHk_prevScan,sHg_prevScan,sHk_prevScan,params,keepOtherScans=False)
        pt2dproc.addNewKeyFrameAndScan(poseGraph,KeyFrame_prevIdx,idx-1,idx,X,idx,
                      params,timeMetrics,keepOtherScans=True)
        et=time.time()
        print("time taken for new keyframe = ",et-st)

        
        KeyFrame_prevIdx = idx-1
        # pt2dplot.plotcomparisons(poseGraph,KeyFrame_prevIdx,idx,UseLC=False,H12=nplinalg.inv( sHk) ,err=serrk)
        # fig = plt.figure("ComparisonPlot")
        # fig.savefig("ComparisonPlot-%d-%d.png"%(KeyFrame_prevIdx, idx))
        # plt.close(fig)
        
        # KeyFrame_prevIdx = idx
        # KeyFrames.append(np.matmul(gHs,np.array([0,0,1]))  )
        
        Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
        Lkeyloop.sort()
            
        # detect loop closure and add the edge
        if params['doLoopClosure'] and np.abs(previdx_loopdetect-idx)>10:
            
            # poseGraph=pt2dproc.detectAllLoopClosures_closebyNodes(poseGraph,params,returnCopy=False,parallel=params['Loop_CLOSURE_PARALLEL'])
            # poseGraph=pt2dproc.LoopCLose_CloseByNodes(poseGraph,params)
            
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
        poseGraph.edges[KeyFrame_prevIdx,idx]['posematch']=posematch
            
        
    
        # pt2dplot.plotcomparisons(poseGraph,KeyFrame_prevIdx,idx,UseLC=False,H12=nplinalg.inv( sHk) ,err=serrk)
        # fig = plt.figure("ComparisonPlot")
        # fig.savefig("ComparisonPlot-%d-%d.png"%(KeyFrame_prevIdx, idx))
        # plt.close(fig)
    


    previdx = idx
    
    
    # plotting
    if idx%50==0 or idx==idxLast-1:
        st = time.time()
        pt2dplot.plot_keyscan_path(poseGraph,idx1,idx,params,makeNew=False,skipScanFrame=True,plotGraphbool=True,
                                    forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True,plotKeyFrameNodesTraj=True)
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

with open("PoseGraph-deutchesMesuemDebug-planes-0p3.pkl",'wb') as fh:
    pkl.dump([poseGraph,params],fh)


    




#%%

with open("PoseGraph-deutchesMesuemDebug-planes-0p3.pkl",'rb') as fh:
    poseGraph,params=pkl.load(fh)

# with open("turtlebot/OKTloopClosed.pkl",'rb') as fh:
#     poseGraph,params,timeMetrics=pkl.load(fh)
    
# for k in timeMetrics:
#     plt.figure(k)
#     plt.hist(timeMetrics[k],bins=100)
    
Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))


GlobalBinMap={'xedges':np.arange(-100,75,0.05),'yedges':np.arange(-75,175,0.05),'H':None}

pt2dplot.plot_keyscan_path(poseGraph,Lkeyloop[0],Lkeyloop[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True,plotKeyFrameNodesTraj=False,CloseUpRadiusPlot=30,GlobalBinMap=None)


Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))

# res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,Lkeys[0],Lkeys[-1],maxiter=None,algo='trf')
# poseGraph=pt2dproc.updateGlobalPoses(poseGraph,sHg_updated,updateRelPoses=True)
            
            

Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))
# Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
# Ledges = poseGraph.edges

# Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
# for nn in Lkeys:
#     if poseGraph.nodes[nn]['clf'] is None or 'clf' not in poseGraph.nodes[nn]:
#         X = poseGraph.nodes[nn]['X']
#         res = pt2dproc2.getclf(X,params,doReWtopt=True,means_init=None)
#         clf=res['clf']
#         poseGraph.nodes[nn]['clf']=clf
        
# for previdx,idx  in poseGraph.edges:
#     dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
#     Xp=poseGraph.nodes[previdx]['X']
#     Xi=poseGraph.nodes[idx]['X']
#     Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc2.binScanEdges(Xp,Xi,dxcomp)
#     activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
#     sHk=poseGraph.edges[previdx,idx]['H']
#     posematch=pt2dproc2.eval_posematch(sHk,Xi,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
#     if posematch['mbinfrac_ActiveOvrlp']<=0.2:
#         posematch2= pt2dproc2.poseGraph_keyFrame_matcher_binmatch(poseGraph,previdx,idx,params,DoCLFmatch=False,dx0=0.5,L0=10,th0=np.pi/3,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
#         posematch2['when']="DoAllLoopClosures-Redo"
#         posematch2['method']='binmatch'
#         print(previdx,idx,posematch['mbinfrac_ActiveOvrlp'],posematch2['mbinfrac_ActiveOvrlp'])
#         if posematch2['mbinfrac_ActiveOvrlp']>posematch['mbinfrac_ActiveOvrlp']:        
#             poseGraph.edges[previdx,idx]['H']=posematch2['H']
#             poseGraph.edges[previdx,idx]['posematchBinMatchRedo']=poseGraph.edges[previdx,idx]['posematch']
#             poseGraph.edges[previdx,idx]['posematch']=posematch2


# poseGraph=pt2dproc2.updated_sHg(poseGraph)
# pt2dplot2.plot_keyscan_path(poseGraph,Lkeyloop[0],Lkeyloop[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
#                                    forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

for i in range(1650,len(Lkeyloop)):
    poseGraph.nodes[Lkeyloop[i]]['LongLoopDonePrevIdxs']=[]
    

pgopt=None

Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
for i in range(0,len(Lkeyloop),10):
    idxlb=Lkeyloop[max([0,i-13])]
    idxub=Lkeyloop[i]
    print(i,idxlb,idxub)
    poseGraph=pt2dproc.detectAllLoopClosures(poseGraph,params,idxlb=idxlb,idxub=idxub,returnCopy=False,parallel=True) #
    Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    st=time.time()        
    poseGraph,pgopt=pt2dproc.adjustPosesG2o(poseGraph,pgopt=None)
    et=time.time()
    print("G2o optimization = ",et-st)
    pt2dplot.plot_keyscan_path(poseGraph,Lkeyloop[0],idxub,params,makeNew=False,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
Lkeyloop_edges=[ee for ee in Lkeyloop_edges if ee[1]>=18000]
poseGraph.remove_edges_from(Lkeyloop_edges)

with open("PoseGraph-deutchesMesuemDebug-planes-0p3-good.pkl",'wb') as fh:
    pkl.dump([poseGraph,params,timeMetrics],fh)

for i in poseGraph.nodes:
    if i<=42516:
        continue
    pt2dplot.plot_keyscan_path(poseGraph,Lkeyloop[0],i,params,makeNew=False,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True,plotKeyFrameNodesTraj=False,CloseUpRadiusPlot=25,GlobalBinMap=None)
    fig = plt.figure("Full Plot")
    fig.savefig("debugPlots/sim_0p3-K2K_0p4-K2KLoop_%06d.png"%(i,))
    
    fig = plt.figure("Close-Up Plot")
    fig.savefig("debugPlots/CloseBysim_0p3-K2K_0p4-K2KLoop_%06d.png"%(i,))
    
    # plt.close(fig)

         
L1=[41313, 41354]
L2=[0,166]
posematch2= pt2dproc.poseGraph_keyFrame_matcher_binmatch(poseGraph,0,41313,params,DoCLFmatch=True,dx0=0.9,L0=2,th0=np.pi/4,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
posematch = pt2dproc.poseGraph_keyFrame_matcher_binmatch(poseGraph,0,41313,params,dx0=params['LongLoopClose']['Bin_Match_dx0'],L0=params['LongLoopClose']['Bin_Match_L0'],th0=params['LongLoopClose']['Bin_Match_th0'],DoCLFmatch=params['LongLoopClose']['DoCLFmatch'],PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)

for previdx,idx  in Lkeyloop_edges:
    # if os.path.isfile("debugPlots/Key2Key-%d-%d_gmm.png"%(idx, previdx)):
    #     continue
    # if idx>=38312:
    #     pass
    # else:
    #     continue
    
    if 'posematch' not in poseGraph.edges[previdx,idx]:
        print("no posematch for %d-%d"%(previdx,idx))
        posematch={'mbinfrac_ActiveOvrlp':-1}
    else:
        posematch=poseGraph.edges[previdx,idx]['posematch']        
    
    # dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    # Xp=poseGraph.nodes[previdx]['X']
    # Xi=poseGraph.nodes[idx]['X']
    # Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc2.binScanEdges(Xp,Xi,dxcomp)
    # activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    # sHk=poseGraph.edges[previdx,idx]['H']
    # posematch=pt2dproc.eval_posematch(sHk,Xi,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
    # piHi=nplinalg.inv(sHk)
    
    
    if posematch['mbinfrac_ActiveOvrlp']<=0.2:
        print(previdx,idx)
        # posematch2= pt2dproc2.poseGraph_keyFrame_matcher_binmatch(poseGraph,previdx,idx,params,DoCLFmatch=True,dx0=0.9,L0=2,th0=np.pi/4,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
        # posematch2=pt2dproc2.poseGraph_keyFrame_matcher_long(poseGraph,previdx,idx,params,params['LongLoopClose']['PoseGrid'],
        #                                                     params['LongLoopClose']['isPoseGridOffset'],
        #                                                     params['LongLoopClose']['isBruteForce'])
        # # mbinfrac=posematch['mbinfrac']
        # mbinfrac_ActiveOvrlp=posematch['mbinfrac_ActiveOvrlp']
        
        # piHi=posematch['H']
        
        # poseGraph.edges[previdx,idx]['H']=posematch2['H']
        # poseGraph.edges[previdx,idx]['posematchGMM']=posematch
        # poseGraph.edges[previdx,idx]['posematch']=posematch2
        
        # kHg = poseGraph.nodes[previdx]['sHg']
        # sHg = np.matmul(posematch2['H'],kHg)
        # gHs=nplinalg.inv(sHg)    
        # tpos=np.matmul(gHs,np.array([0,0,1])) 
        # poseGraph.nodes[idx]['pos']=(tpos[0],tpos[1])
        # poseGraph.nodes[idx]['sHg']=sHg

        piHi=nplinalg.inv(poseGraph.edges[previdx,idx]['H'])
        # piHi=poseGraph.edges[previdx,idx]['posematch']['H']
            
        pt2dplot2.plotcomparisons(poseGraph,previdx,idx,UseLC=False,H12=piHi,err=posematch['mbinfrac_ActiveOvrlp']) #nplinalg.inv(piHi) 
        fig = plt.figure("ComparisonPlot")
        fig.savefig("debugPlots/Key2Key-%d-%d_gmm.png"%(idx, previdx))
        plt.pause(0.1)
        plt.close(fig)
        
        # piHi=nplinalg.inv(posematch2['H'])
        # pt2dplot2.plotcomparisons(poseGraph,previdx,idx,UseLC=False,H12=piHi,err=posematch2['mbinfrac_ActiveOvrlp']) #nplinalg.inv(piHi) 
        # fig = plt.figure("ComparisonPlot")
        # fig.savefig("debugPlots/Key2Key-%d-%d_bin.png"%(idx, previdx))
        # plt.close(fig)
        
        # break
    

poseGraph=pt2dproc2.updated_sHg(poseGraph)

pt2dplot2.plot_keyscan_path(poseGraph,Lkeyloop[0],Lkeyloop[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

pt2dplot2.plot_keyscan_path(poseGraph,30387,Lkeyloop[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

#%%
import heapq
with open("DeutchesMeuseum_g2oTest_good2.pkl",'rb') as fh:
    poseGraph,params,_=pkl.load(fh)

class CostAndNode:
    def __init__(self, cost, node):
        self.cost = cost
        self.node = node

    # do not compare nodes
    def __lt__(self, other):
        return self.cost < other.cost
    


plt.close("all")
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc,boolean
from numba import njit, prange,jit

# @jit
def getPointCost(H,dx,X,Oj,Tj):
    # Tj is the 2D index of displacement
    # X are the points
    # dx is 2D
    # H is the probability histogram
    c=0
    P=np.floor(X/dx).astype(int)
    j=np.floor(Oj/dx).astype(int)
    
    Pn=P+j

    
    idx=np.prod(np.logical_and(Pn>=np.zeros(2) , Pn<H.shape),axis=1 ).astype(bool)
    Pn=Pn[idx,:]

    if Pn.size>0:
        c=np.sum(H[Pn[:,0],Pn[:,1]])
    return c


def UpsampleMax(Hup,n):
    H=np.zeros((int(np.ceil(Hup.shape[0]/2)),int(np.ceil(Hup.shape[1]/2))))
    for j in range(H.shape[0]):
        for k in range(H.shape[1]):
            lbx=max([2*j,0])
            ubx=min([2*j+n,Hup.shape[0]-1])+1
            lby=max([2*k,0])
            uby=min([2*k+n,Hup.shape[1]-1])+1
            H[j,k] = np.max( Hup[lbx:ubx,lby:uby] )
    return H

import numpy as np
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
# H=pool2d(Hup, kernel_size=3, stride=2, padding=0, pool_mode='max')



def binMatcherAdaptive(X11,X22,H12,Lmax,thmax,dxMatch,dxMax):
    # dxMax is the max resolution allowed
    # Lmax =[xmax,ymax]
    # search window is [-Lmax,Lmax] and [-thmax,thmax]
    n=histsmudge =2 # how much overlap when computing max over adjacent hist for levels
    
    
    mn=np.zeros(2)
    mx=np.zeros(2)
    mn_orig=np.zeros(2)
    mn_orig[0] = np.min(X11[:,0])
    mn_orig[1] = np.min(X11[:,1])
    
    R=H12[0:2,0:2]
    t=H12[0:2,2]
    X222 = R.dot(X22.T).T+t
    
    
    X2=X222-mn_orig
    X1=X11-mn_orig
    
    # print("mn_orig = ",mn_orig)
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    rmax=np.max(np.sqrt(X2[:,0]**2+X2[:,1]**2))
    
    
    # print("mn,mx=",mn,mx)
    P = mx-mn
    
    
    dxMax[0] = np.min([dxMax[0],Lmax[0]/2,P[0]/2])
    dxMax[1] = np.min([dxMax[1],Lmax[1]/2,P[1]/2])
    
    nnx=np.ceil(np.log2(P[0]))
    nny=np.ceil(np.log2(P[1]))
    
    xedges=np.arange(mn[0]-dxMatch[0],mx[0]+dxMax[0],dxMatch[0])
    yedges=np.arange(mn[1]-dxMatch[0],mx[1]+dxMax[0],dxMatch[1])
    
    if len(xedges)%2==0:
        xedges=np.hstack([xedges,xedges[-1]+1*dxMatch[0]])
    if len(yedges)%2==0:
        yedges=np.hstack([yedges,yedges[-1]+1*dxMatch[1]])
        
    
    H1match=nbpt2Dproc.numba_histogram2D(X1, xedges,yedges)
    
    H1match[H1match>1]=1
    
    # H1match=H1match/(np.sum(H1match)*np.prod(dxMatch))
    thfineRes = np.max([0.5*np.min(dxMatch)/rmax,2*np.pi/180])
    
    
    
    # first create multilevel histograms
    levels=[]
    HLevels=[H1match]
    dxs = [dxMatch]
    # serachboxX,serachboxY=np.meshgrid(np.arange(-Lmax[0],Lmax[0]+dxMatch[0],dxMatch[0]),np.arange(-Lmax[1],Lmax[1]+dxMatch[1],dxMatch[1]))
    SolBoxes=[]
    S=[]
    for xs in np.arange(-Lmax[0],Lmax[0],dxMatch[0]):
        for ys in np.arange(-Lmax[1],Lmax[1],dxMatch[1]):
            S.append( (np.array([xs,ys]),dxMatch) )
    SolBoxes.append(S)
    # XYedges=[(xedges,yedges)]
    flg=False
    st=time.time()
    for i in range(1,100):
        
        dx=2*dxs[i-1]
        if np.any(dx>dxMax):
            flg=True
        
        Hup = HLevels[i-1]
        H=pool2d(Hup, kernel_size=3, stride=2, padding=0, pool_mode='max')
        # H=UpsampleMax(Hup,n)
        
        # print(xedges[0],xedges[-1],len(xedges),yedges[0],yedges[-1],len(yedges))
        # lx =xedges[-1]
        # ly =yedges[-1]
        # xedges=xedges[::2]
        # yedges=yedges[::2]
        # print(xedges[0],xedges[-1],len(xedges),yedges[0],yedges[-1],len(yedges))
        # print("-------------")
        # pt2dproc.plotbins2(xedges,yedges,H,X1,X2)
        # XYedges.append((xedges,yedges))  
        # if len(xedges)%2==0:
        #     xedges=np.hstack([xedges,lx])
        # if len(yedges)%2==0:
        #     yedges=np.hstack([yedges,ly])
            
        HLevels.append(H)
        dxs.append(dx)
          
        # S=[]
        # for xs in np.arange(-Lmax[0],Lmax[0],dx[0]):
        #     for ys in np.arange(-Lmax[1],Lmax[1],dx[1]):
        #         S.append( (np.array([xs,ys]),dx) )
        # SolBoxes.append(S)
        if flg:
            break
    HLevels=HLevels[::-1]
    dxs=dxs[::-1]
    # XYedges=XYedges[::-1]
    # SolBoxes=SolBoxes[::-1]
    et=time.time()
    print("Time pre-init = ",et-st)
    SolBoxes_init=[]
    for xs in np.arange(-Lmax[0],Lmax[0]+1.5*dxs[0][0],dxs[0][0]):
        for ys in np.arange(-Lmax[1],Lmax[1]+1.5*dxs[0][1],dxs[0][1]):
            SolBoxes_init.append( (np.array([xs,ys]),dxs[0]) )
    
    
    mxLVL=len(HLevels)-1
    
    # ff=open("debugPlots/DebugCosts.txt","w")
    # cc=0
    st=time.time()
    decimatedict={}
    h=[]
    #Initialize with all thetas fixed at Max resolution
    lvl=0
    dx=dxs[lvl]
    H=HLevels[lvl]
    Xth={}
    thL=np.arange(-thmax,thmax+thfineRes,thfineRes)
    # np.random.shuffle(thL)
    for th in thL:
        R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        XX=np.transpose(R.dot(X2.T))
        Xth[th]=XX[XX[:,0].argsort()]
        
        
        for solbox in SolBoxes_init:
            Tj=solbox[1]
            Oj = solbox[0]
            if not np.all(Tj==dx):
                print(Tj,dx)
                raise Exception("Tj and dx are not equal ")
                
            
            cost2=getPointCost(H,dx,Xth[th],Oj,Tj)
            # print(-cost2,[solbox,lvl,th])
            heapq.heappush(h,CostAndNode(-cost2,[solbox,lvl,th]))
            # heapq.heappush(h,(-cost2,(lvl,th,solbox)))
            # if lvl>=4:
            # fig=plt.figure("a")
            # ax=fig.add_subplot(111)
            # ax.pcolormesh(XYedges[lvl][0],XYedges[lvl][1],HLevels[lvl].T,shading='flat',alpha=0.4 )
            # ax.plot(X1[:,0],X1[:,1],'r.',label="hist points")
            # ax.plot(Xth[th][:,0]+solbox[0][0],Xth[th][:,1]+solbox[0][1],'b.',label="matching points")
            # ax.axis('equal')
            # ax.set_title(str(cost2)+" "+str(0))
            # ax.legend()
            # fig.savefig("debugPlots/%d.png"%(cc,))
            # ff.write(str(cc)+"   "+str((cost2,[solbox,0,th]))+"\n")
            # cc+=1    
            # plt.cla()
    et=time.time()
    print("Time post-init = ",et-st)        
    
    st=time.time()
    while(1):
        CN=heapq.heappop(h)
        (cost,[solboxt,lvl,th])=(CN.cost,CN.node)
        # if lvl>=4:
        # fig=plt.figure("a")
        # ax=fig.add_subplot(111)
        # ax.pcolormesh(XYedges[lvl][0],XYedges[lvl][1],HLevels[lvl].T,shading='flat',alpha=0.4 )
        # ax.plot(X1[:,0],X1[:,1],'r.',label="hist points")
        # ax.plot(Xth[th][:,0]+solboxt[0][0],Xth[th][:,1]+solboxt[0][1],'b.',label="matching points")
        # ax.axis('equal')
        # ax.set_title(str(cost)+" "+str(lvl))
        # ax.legend()
        # fig.savefig("debugPlots/%d.png"%(cc,))
        # ff.write(str(cc)+"   "+str((cost,[solboxt,lvl,th]))+"\n")
        # cc+=1    
        # plt.cla()
        
        
        # print("----------")
        # print(cost,lvl,mxLVL,th)
        if lvl==mxLVL:
            print("done")
            break
        dx=dxs[lvl+1]
        H=HLevels[lvl+1]
        Tj=solboxt[1]
        Oj=solboxt[0]
        S=[]
        # print(solboxt,dx)
        Xg=np.arange(Oj[0],Oj[0]+Tj[0],dx[0])
        Yg=np.arange(Oj[1],Oj[1]+Tj[1],dx[1])
        for xs in Xg[:2]:
            for ys in Yg[:2]:
                S.append( (np.array([xs,ys]),dx) )
        # print(len(S))
        # print(S)        
        for solbox in S:
            Tj=solbox[1]
            Oj = solbox[0]
            # if not np.all(Tj==dx):
            #     raise Exception("Tj and dx are not equal ")
            cost=getPointCost(H,dx,Xth[th],Oj,Tj)
            heapq.heappush(h,CostAndNode(-cost,[solbox,lvl+1,th]))
    
    et=time.time()
    print("heap pops = ",et-st)
    # ff.close()
    
    t=solboxt[0]
    H=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    H[0:2,0:2]=R
    H[0:2,2]=t
    # print(H,R,t)
    # XX=R.dot(X2.T).T+t
    # plt.figure("check 1")
    # plt.plot(X1[:,0],X1[:,1],'b.')
    # plt.plot(XX[:,0],XX[:,1],'r.')
    
    # plt.figure("check 2")
    # plt.plot(X1[:,0],X1[:,1],'b.')
    # plt.plot(Xth[th][:,0]+t[0],Xth[th][:,1]+t[1],'r.')
    
    # H=nbpt2Dproc.getHmat(th,t)
    Htotal12 = np.matmul(H,H12)
    RT=Htotal12[0:2,0:2]
    tT=Htotal12[0:2,2]
    
    Rs=H[0:2,0:2]
    ts=H[0:2,2]
    
    t = tT-(Rs.dot(mn_orig)+0*ts)+mn_orig
    Htotal12_updt=Htotal12.copy()
    Htotal12_updt[0:2,2]=t
    Htotal21_updt = nplinalg.inv(Htotal12_updt)
    return Htotal21_updt,cost

Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))
e1=Lkeyloop_edges[1][0]
e2=Lkeyloop_edges[1][1]
X1=poseGraph.nodes[e1]['X']
X2=poseGraph.nodes[e2]['X']
H21=np.identity(3) 
th=0*np.pi/180
R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
H21[0:2,0:2]=R
# H21=poseGraph.edges[e1,e2]['H']
# H21[0:2,2]=H21[0:2,2]+5
H12 = nplinalg.inv(H21)
Lmax=np.array([7,7])
thmax=25*np.pi/180
dxMatch=np.array([0.2,0.2])
dxMax=np.array([4,4])
st=time.time()
Hbin21,cost=binMatcherAdaptive(X1,X2,H12,Lmax,thmax,dxMatch,dxMax)
et=time.time()
X1=poseGraph.nodes[e1]['X']
X2=poseGraph.nodes[e2]['X']
print("Best match = ",cost,Hbin21, " in time = ",et-st)
Hbin12 = nplinalg.inv(Hbin21)
R=Hbin12[0:2,0:2]
t=Hbin12[0:2,2]
X22 = R.dot(X2.T).T+t

plt.figure()
plt.plot(X1[:,0],X1[:,1],'b.')
plt.plot(X22[:,0],X22[:,1],'r.')


#%% g20 testing
with open("DeutchesMeuseum_g2oTest.pkl",'rb') as fh:
    poseGraph,params,timeMetrics=pkl.load(fh)


import numpy
import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=50):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=g2o.RobustKernelHuber()):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
G = poseGraph

def adjustPosesG2o(poseGraph,pgopt=None):
    if pgopt is None:
        pgopt=PoseGraphOptimization()
    # nodeList = [nn for nn in poseGraph.nodes if nn>=idx0 and nn<=idx1]
    # nodeList.sort()
    
    for nn in poseGraph.nodes:
        if poseGraph.nodes[nn]['frametype']!="keyframe":
            continue
        
        if pgopt.vertex(nn) is not None:
            continue
        
        gHs=nplinalg.inv(poseGraph.nodes[nn]['sHg'])
        R=np.identity(3)
        R[0:2,0:2]=gHs[0:2,0:2]
        tpos=np.zeros(3)
        tpos[0:2]=gHs[0:2,2]
        
        if nn==0:
            pgopt.add_vertex(nn,g2o.Isometry3d(R, tpos),fixed=True)
        else:
            pgopt.add_vertex(nn,g2o.Isometry3d(R, tpos),fixed=False)
    
    L=list(pp.edges())
    L=map(lambda x: x.vertices(),L)
    edgeList=list(map(lambda x: (x[0].id(),x[1].id()),L))
    
    for (e1,e2) in poseGraph.edges:
        
        
        # if max([e1,e2])<idx0 or min([e1,e2])>idx1:
        #     continue
        if (e1,e2) in edgeList:
            continue
        if poseGraph.edges[e1,e2]['edgetype']=="Key2Key" or poseGraph.edges[e1,e2]['edgetype']=="Key2Key-LoopClosure":
            pass
        else:
            continue
        
        H=nplinalg.inv(poseGraph.edges[e1,e2]['H'])
        R=np.identity(3)
        R[0:2,0:2]=H[0:2,0:2]
        tpos=np.zeros(3)
        tpos[0:2]=H[0:2,2]
        
        err = poseGraph.edges[e1,e2]['posematch']['mbinfrac_ActiveOvrlp'] 
        Hess = np.identity(6)
        Hess[0:3,0:3]=1*poseGraph.edges[e1,e2]['hess_inv']
        pgopt.add_edge([e1,e2],g2o.Isometry3d(R, tpos),information=Hess)
    
    
    pgopt.optimize()
    
    for nn in poseGraph.nodes:   
        if poseGraph.nodes[nn]['frametype']!="keyframe":
            continue
        
        v1=pgopt.get_pose(nn)
        tpos=v1.position()
        R = v1.rotation_matrix()
        H=np.identity(3)
        H[0:2,0:2]=R[0:2,0:2]
        H[0:2,2]=tpos[0:2]
        
        poseGraph.nodes[nn]['sHg']=nplinalg.inv(H)
        poseGraph.nodes[nn]['pos']=(tpos[0],tpos[1])
    

    for ns in list(poseGraph.nodes):
        # if ns<=nodeList[-1] and ns>=nodeList[0]:            
        for pidx in poseGraph.predecessors(ns):
            if poseGraph.nodes[pidx]['frametype']=="keyframe": # and pidx in sHg_updated
                if poseGraph.edges[pidx,ns]['edgetype']=="Key2Scan":
                    psHg=poseGraph.nodes[pidx]['sHg']
                    nsHps=poseGraph.edges[pidx,ns]['H']
                    nsHg = nsHps.dot(psHg)
                    poseGraph.nodes[ns]['sHg']=nsHg
                    gHns=nplinalg.inv(nsHg)
                    tpos=np.matmul(gHns,np.array([0,0,1]))
                    poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
                    break
                    
        # if ns>nodeList[-1]:
        # for pidx in poseGraph.predecessors(ns):
        #     if poseGraph.nodes[pidx]['frametype']=="keyframe": # and pidx in sHg_updated
        #         if poseGraph.edges[pidx,ns]['edgetype']=="Key2Key" or poseGraph.edges[pidx,ns]['edgetype']=="Key2Scan":
        #             psHg=poseGraph.nodes[pidx]['sHg']
        #             nsHps=poseGraph.edges[pidx,ns]['H']
        #             nsHg = nsHps.dot(psHg)
        #             poseGraph.nodes[ns]['sHg']=nsHg
        #             gHns=nplinalg.inv(nsHg)
        #             tpos=np.matmul(gHns,np.array([0,0,1]))
        #             poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
        #             break
    return poseGraph,pgopt


pp=PoseGraphOptimization()
for nn in G.nodes:
    gHs=nplinalg.inv(G.nodes[nn]['sHg'])
    R=np.identity(3)
    R[0:2,0:2]=gHs[0:2,0:2]
    tpos=np.zeros(3)
    tpos[0:2]=gHs[0:2,2]
    if nn==0:
        pp.add_vertex(nn,g2o.Isometry3d(R, tpos),fixed=True)
    else:
        pp.add_vertex(nn,g2o.Isometry3d(R, tpos),fixed=False)


for (e1,e2) in G.edges:
    H=nplinalg.inv(G.edges[e1,e2]['H'])
    R=np.identity(3)
    R[0:2,0:2]=H[0:2,0:2]
    tpos=np.zeros(3)
    tpos[0:2]=H[0:2,2]
    
    Hess = np.identity(6)
    Hess[0:3,0:3]=G.edges[e1,e2]['hess_inv']
    pp.add_edge([e1,e2],g2o.Isometry3d(R, tpos),information=Hess)


pp.optimize()

G2=copy.deepcopy(G)

for nn in G2.nodes:
    
    v1=pp.get_pose(nn)
    tpos=v1.position()
    R = v1.rotation_matrix()
    H=np.identity(3)
    H[0:2,0:2]=R[0:2,0:2]
    H[0:2,2]=tpos[0:2]
    
    G2.nodes[nn]['sHg']=nplinalg.inv(H)
    G2.nodes[nn]['pos']=(tpos[0],tpos[1])

Gkey = list(filter(lambda x: G2.nodes[x]['frametype']=="keyframe",G2.nodes))

pt2dplot2.plot_keyscan_path(G2,Gkey[0],Gkey[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

 
#%%
