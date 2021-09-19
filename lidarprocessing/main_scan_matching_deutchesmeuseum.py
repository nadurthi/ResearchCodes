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
        intelfile = "lidarprocessing/datasets/Intel Research Lab.clf"
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
        ths=np.arange(0,90*np.pi/180,0.5*np.pi/180)
        for i in range(cnt):
            rngs.append(float(glist[2+i]))
        rngs = np.array(rngs)
        p=np.vstack([np.cos(ths),np.sin(ths)])
        ptset = rngs.reshape(-1,1)*p.T
        
        return ptset
dataset = IntelData()       
def getscanpts_intel(dataset,idx):
    # ranges = dataset[i]['ranges']
   
    X=dataset.getflaser(idx)
    
    # now filter silly points
    tree = KDTree(X, leaf_size=5)
    cnt = tree.query_radius(X, 0.25,count_only=True) 
    X = X[cnt>=2,:]
    
    cnt = tree.query_radius(X, 0.5,count_only=True) 
    X = X[cnt>=5,:]
    
    return X

getscanpts = getscanpts_deutches
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
params['BinDownSampleKeyFrame_probs']=0.1

params['Plot_BinDownSampleKeyFrame_dx']=0.05
params['Plot_BinDownSampleKeyFrame_probs']=0.01

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
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']=0.3
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
                      params,timeMetrics,keepOtherScans=False)
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


    




#%%

with open("DeutchesMeuseum_g2oTest_opt2.pkl",'rb') as fh:
    poseGraph,params,_=pkl.load(fh)

# with open("turtlebot/OKTloopClosed.pkl",'rb') as fh:
#     poseGraph,params,timeMetrics=pkl.load(fh)
    
# for k in timeMetrics:
#     plt.figure(k)
#     plt.hist(timeMetrics[k],bins=100)
    
Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))

pt2dplot.plot_keyscan_path(poseGraph,Lkeyloop[0],Lkeyloop[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)


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
for i in range(1702,Lkeyloop[-1],10):
    idxlb=Lkeyloop[max([0,i-13])]
    idxub=Lkeyloop[i]
    print(i,idxlb,idxub)
    poseGraph=pt2dproc.detectAllLoopClosures(poseGraph,params,idxlb=idxlb,idxub=idxub,returnCopy=False,parallel=True) #
    Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    st=time.time()        
    poseGraph,pgopt=adjustPosesG2o(poseGraph,pgopt=None)
    et=time.time()
    print("G2o optimization = ",et-st)
    pt2dplot.plot_keyscan_path(poseGraph,Lkeyloop[0],idxub,params,makeNew=False,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)

Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
Lkeyloop_edges=[ee for ee in Lkeyloop_edges if ee[1]>=18000]
poseGraph.remove_edges_from(Lkeyloop_edges)

with open("DeutchesMeuseum_g2oTest_opt2.pkl",'wb') as fh:
    pkl.dump([poseGraph,params,timeMetrics],fh)
          
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
def binMatcherAdaptive(X1,X22,H12,Lmax,thmax,dxMatch,dxMax):
    # dxMax is the max resolution allowed
    # search window is [-Lmax,Lmax] and [-thmax,thmax]
    R=H12[0:2,0:2]
    t=H12[0:2,2]
    X2 = R.dot(X22).T+t
    X2=X2.transpose()
    
    rmax=np.max(np.sqrt(X2[:,0]**2+X2[:,1]**2))
    
    mn=np.zeros(2)
    mx=np.zeros(2)
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    
    dxMax = np.min([dxMax,(mx[0]-mn[0])/2,(mx[1]-mn[1])/2])
    
    xedges=np.arange(mn[0]-dxMatch,mx[0]+dxMatch,dxMatch)
    yedges=np.arange(mn[1]-dxMatch,mx[1]+dxMatch,dxMatch)
    if len(xedges)%2!=0:
        xedges=np.hstack([xedges,mx[0]+1*dxMatch)
    if len(yedges)%2!=0:
        yedges=np.hstack([yedges,mx[1]+1*dxMatch)
        
    H1match=nbpt2Dproc.numba_histogram2D(X1, xedges,yedges)
    thfineRes = 0.5*dxMatch/rmax
    
    # first create multilevel histograms
    HLevels=[[dxMatch,H1match,xedges,yedges]]
    
    dx=dxMatch
    for i in range(1,100):
        dx=2*dx
        if dx>dxMax:
            break
        
        
        Hup = HLevels[i-1][0]
        xedgesup=HLevels[i-1][1]
        yedgesup=HLevels[i-1][2]
        
        M, N = Hup.shape
        K = 2
        L = 2
        
        MK = M // K
        NL = N // L
        H=Hup[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3))
        HLevels.append([dx,H,xedgesup[::2],yedgesup[::2]])

    maxLevel = len(HLevels)-1
    h=[]
    #Initialize with all thetas fixed at Max resolution
    for th in np.arange(-thmax,thmax+thfineRes,thfineRes):
        
        heapq.heappush(h,(cost,[lvl,th,solboxIdx]))
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
