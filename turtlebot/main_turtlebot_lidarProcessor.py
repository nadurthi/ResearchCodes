# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:46:31 2020

@author: nadur
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data, qos_profile_system_default
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan, Imu, MultiEchoLaserScan
from std_msgs.msg import String
# from sensor_msgs.msg import LaserScan, Imu 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose,PoseStamped

# import multiprocessing as mp
# import threading
# import queue


import pickle
import signal
import datetime

import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.neighbors import KDTree
from uq.gmm import gmmfuncs as uqgmmfnc
from utils.plotting import geometryshapes as utpltgmshp
import time
from scipy.optimize import minimize, rosen, rosen_der,least_squares
from scipy import interpolate
from scipy import linalg as sclinalg
import networkx as nx
import pdb
import pandas as pd
from fastdist import fastdist
import copy
from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point2Dplotting as pt2dplot
import codecs
import turtlebot_helper as ttlhelp  
# from turtlebot_helper import params
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
from threading import Thread, Lock
#https://quaternion.readthedocs.io/en/latest/
import quaternion



import datetime
dtype = np.float64


from rclpy.duration import Duration
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSLivelinessPolicy
from rclpy.qos import QoSPresetProfiles
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy



 #%%
# Terminology:
    # - GMM or gmm or clf or classifier: gaussian mixture classifer that fits the point cloud
    # - Keyframe contains the gmm classifier
    # - subsequent scans are matched to the previous keyframe and pose is estimated to this key frame
    # - gHs : scanframe to global frame
    # - kHs : scanframe to keyframe
    # - sHk : keyframe to scanframe
    # - for posegraph: nodes carry the sHg transformation matrix from global to current frame
    #                : edges carry the H matrix transformation between the frames.
    
    
    
params={}

params['REL_POS_THRESH']=0.25# meters after which a keyframe is made
params['REL_ANGLE_THRESH']=15*np.pi/180
params['ERR_THRES']=2.3
params['n_components']=25
params['reg_covar']=0.002

params["Key2Key_Overlap"]=0.5
params["Scan2Key_Overlap"]=0.5

params['BinDownSampleKeyFrame_dx']=0.1
params['BinDownSampleKeyFrame_probs']=0.0001

params['Plot_BinDownSampleKeyFrame_dx']=0.05
params['Plot_BinDownSampleKeyFrame_probs']=0.001

params['doLoopClosure'] = True
params['doLoopClosureLong'] = True

params['Loop_CLOSURE_PARALLEL'] = True
params['LOOP_CLOSURE_D_THES']=31.4
params['LOOP_CLOSURE_POS_THES']=10
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
# params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
# params['LOOPCLOSE_BIN_MATCHER_L'] = 13
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = np.array([0.1,0.1],dtype=np.float64)

params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.2
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']=0.6
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']=0.7
params['LOOP_CLOSURE_COMBINE_MAX_NODES']= 8

params['offsetNodesBy'] = 2


params['MAX_NODES_ADJ_COMBINE']=5
params["USE_Side_Combine"]=False
params["Side_Combine_Overlap"]=0.6


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
params['LongLoopClose']['Bin_Match_dx0'] = 0.5
params['LongLoopClose']['Bin_Match_L0'] = 2
params['LongLoopClose']['Bin_Match_th0'] = np.pi/4
params['LongLoopClose']['DoCLFmatch'] = True

params['LongLoopClose']['AlongPathNearFracCountNodes'] = 0.3
params['LongLoopClose']['AlongPathNearFracLength'] = 0.3
params['LongLoopClose']['#TotalRandomPicks'] = 10
params['LongLoopClose']['AdjSkipList'] = 5

# params['Do_GMM_FINE_FIT']=False

# params['Do_BIN_FINE_FIT'] = False

params['Do_BIN_DEBUG_PLOT-dx']=False
params['Do_BIN_DEBUG_PLOT']= False

params['xy_hess_inv_thres']=100000000*0.4
params['th_hess_inv_thres']=100000000*0.4


params['#ThreadsLoopClose']=8

params['INTER_DISTANCE_BINS_max']=120
params['INTER_DISTANCE_BINS_dx']=1


params['LOOPCLOSE_AFTER_#KEYFRAMES']=5

poseData={}
# mutex = mp.Lock()

class ProcessLidarData:
    def __init__(self):
        self.poseGraph = nx.DiGraph()
        self.idx0 = 0
        self.loopClosedFrameidx = 0
        self.poseGraph_closed=None
        
    def setlidarpose_publisher(self,lidarpose_pub):
        self.lidarpose_pub = lidarpose_pub
        
    def setlidarposegraph_publisher(self,lidarposegraph_pub):
        self.lidarposegraph_pub = lidarposegraph_pub
    
    def setplotpose_publisher(self,plotpose_pub):
        self.plotpose_pub = plotpose_pub
        
    def publishPoseGraph(self):
        msg=String()
        # we need to use this weird codecs to encode non-ASCI characters in the pickled string
        # ros does not accept non-ASCI characters. 
        pickled = codecs.encode(pkl.dumps([self.poseGraph,params]), "base64").decode()
        msg.data = pickled
        self.lidarposegraph_pub.publish(msg)
    
    def publishPlotPose(self):
        msg=String()
        # we need to use this weird codecs to encode non-ASCI characters in the pickled string
        # ros does not accept non-ASCI characters. 
        pickled = codecs.encode(pkl.dumps([self.poseGraph,params]), "base64").decode()
        msg.data = pickled
        self.plotpose_pub.publish(msg)
        
    def publishPose(self,Tstamp,gHs):
        tpos=np.matmul(gHs,np.array([0,0,1]))
        msg = PoseStamped()
        msg.header.stamp = Tstamp
        
        msg.pose.position.x = tpos[0]
        msg.pose.position.y = tpos[1]
        msg.pose.position.z = 0.0
        gHs_33 = sclinalg.block_diag(gHs[0:2,0:2],1)
        # print(gHs_33)
        q=quaternion.from_rotation_matrix(gHs_33, nonorthogonal=False)
        q = quaternion.as_float_array(q)
        
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1] 
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        
        self.lidarpose_pub.publish(msg)
        
        
    def processScanPts(self,scanmsg):
        Tstamp,T,X = ttlhelp.filter_scanmsg(scanmsg)
        self.computePose(Tstamp,T,X)
        # return X


    def computePose(self,Tstamp,T,X):
        # mutex.acquire()
        self.setUpdatedPoseGraph()
        
        # first frame is automatically a keyframe
        if len(self.poseGraph.nodes)==0:
            res = pt2dproc.getclf(X,params,doReWtopt=True,means_init=None)
            clf=res['clf']
            H=np.hstack([np.identity(2),np.zeros((2,1))])
            H=np.vstack([H,[0,0,1]])
            idbmx = params['INTER_DISTANCE_BINS_max']
            idbdx=params['INTER_DISTANCE_BINS_dx']
            # h=pt2dproc.get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
            h = np.array([0,0])
            self.idx=self.idx0
            # saving the data X can be removed later when we revise the code. Saving X might consume time
            self.poseGraph.add_node(self.idx,X=X,frametype="keyframe",clf=clf,time=T,sHg=H,pos=(0,0),h=h,color='g',LoopDetectDone=False)
            
            self.KeyFrame_prevIdx=self.idx
            
            self.idx+=1
            
            return True

        # estimate pose to last keyframe
        KeyFrameClf = self.poseGraph.nodes[self.KeyFrame_prevIdx]['clf']
        Xclf = self.poseGraph.nodes[self.KeyFrame_prevIdx]['X']
        
        # get a initial guess for pose optimization using prev frame pose. 
        if (self.idx-self.KeyFrame_prevIdx)<=1: # if it is too close use Idenity as initial guess of pose
            sHk_prevframe = np.identity(3)
        else: # or else get the H pose matrix between prev frame and prev scan frame
            sHk_prevframe = self.poseGraph.edges[self.KeyFrame_prevIdx,self.idx-1]['H']

        # assuming sHk_prevframe is very close to sHk, use it as a guess for pose optimization
        # now match the point cloud X to the previous keyframe
        st=time.time()
        sHk,serrk,shessk_inv = pt2dproc.scan2keyframe_match(KeyFrameClf,Xclf,X,params,sHk=sHk_prevframe)
        et = time.time()
                
        print("idx = ",self.idx," Error = ",serrk," , and time taken = ",et-st)
        # now get the global pose to the frame
        kHg = self.poseGraph.nodes[self.KeyFrame_prevIdx]['sHg'] #global pose to the prev keyframe
        sHg = np.matmul(sHk,kHg) # global pose to the current frame: global to current frame
        gHs=nplinalg.inv(sHg) # current frame to global


        # check if you have to make this the keyframe
        # also if the frame id more than 100 frame away, make it a keyframe
        # when you make it a keyframe, compute the gmm classfier for it to be used in subsequent matchings
        tprevK,thprevK = nbpt2Dproc.extractPosAngle(kHg)
        tcurr,thcurr = nbpt2Dproc.extractPosAngle(sHg)
        thdiff = np.abs(nbpt2Dproc.anglediff(thprevK,thcurr))
        # print("thdiff = ",thdiff)
        # check if to make this the keyframe
        if serrk>params['ERR_THRES'] or nplinalg.norm(sHk[:2,2])>params['REL_POS_THRESH'] or (self.idx-self.KeyFrame_prevIdx)>100 or thdiff>params['REL_ANGLE_THRESH']:
            # make the current idx as the previous keyframe 
            

            print("New Keyframe")
            st = time.time()
            pt2dproc.addNewKeyFrame(self.poseGraph,KeyFrameClf,Xclf,X,self.idx,self.KeyFrame_prevIdx,sHk,sHg,sHk_prevframe,params,keepOtherScans=False)
            et=time.time()
            print("time taken for new keyframe = ",et-st)
            
            
            self.KeyFrame_prevIdx = self.idx

            
            
                
        else: #not a keyframe
            # add the scan frame 
            tpos=np.matmul(gHs,np.array([0,0,1]))
            # saving the data X can be removed later when we revise the code. Saving X might consume time
            self.poseGraph.add_node(self.idx,frametype="scan",time=T,X=X,sHg=sHg,pos=(tpos[0],tpos[1]),color='r',LoopDetectDone=False)
            self.poseGraph.add_edge(self.KeyFrame_prevIdx,self.idx,H=sHk,H_prevframe=sHk_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Scan",color='r')
            
            dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
            Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(Xclf,X,dxcomp)
            activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
            H12=self.poseGraph.edges[self.KeyFrame_prevIdx,self.idx]['H']
            posematch=pt2dproc.eval_posematch(H12,X,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
            posematch['method']='GMMmatch'
            self.poseGraph.edges[self.KeyFrame_prevIdx,self.idx]['posematch']=posematch
            
            
            
        # mutex.release()
        
        self.idx+=1
        self.publishPose(Tstamp,gHs)
        
        if self.idx%50==0:
            self.publishPlotPose()
            pass
        
        # request a loop closure after every say 50 frames
        # when we publish the posegraph on topic "posegraphclose", 
        # the other node with do the loop closure and send it back on topic  "lidarLoopClosedPoses" with callback "updatePoseGraph" below
        Lkey = list(filter(lambda x: self.poseGraph.nodes[x]['frametype']=="keyframe",self.poseGraph.nodes))
        Lkey.sort()
        ilast = Lkey[-1]
        # print("********* %d **********  %d ********* %d "%(ilast,self.loopClosedFrameidx,params['LOOPCLOSE_AFTER_#KEYFRAMES'] ))
        if Lkey.index(ilast)-Lkey.index(self.loopClosedFrameidx) > params['LOOPCLOSE_AFTER_#KEYFRAMES'] :
            self.loopClosedFrameidx = ilast
            print("publish posegraph to call for loop closure")
            self.publishPoseGraph()
             # with lots of frames saving the points X, the posegrph will be large
             # for now let it be, but later we can think of saving the scan points in a separate node/server
    
    def setUpdatedPoseGraph(self):
        
        
        if self.poseGraph_closed is not None:
            
            self.poseGraph=pt2dproc.updateGlobalPoses(copy.deepcopy(self.poseGraph),self.sHg_updated,updateRelPoses=False)
            
            for nn in self.poseGraph_closed.edges:
                if nn not in self.poseGraph.edges:
                    # print("added edge: ",nn,nn[0] in self.poseGraph.nodes,nn[1] in self.poseGraph.nodes)
                    self.poseGraph.add_edge(nn[0],nn[1])
                    
                    for k,v in  self.poseGraph_closed.edges[nn].items():
                        self.poseGraph.edges[nn][k]=v
                elif self.poseGraph_closed.edges[nn].get('modified',False) is True:
                    for k,v in  self.poseGraph_closed.edges[nn].items():
                        if k in self.edge_modified_fields:
                            self.poseGraph.edges[nn][k]=v
            
            for nn in self.poseGraph.edges:
                self.poseGraph.edges[nn]['modified']=False
            
            for nn in self.poseGraph_closed.nodes:
                if nn not in self.poseGraph.nodes:
                    self.poseGraph.add_node(nn)
                    print("added new node: ",nn)
                    for k,v in  self.poseGraph_closed.nodes[nn].items():
                        if k!='X':
                            self.poseGraph.nodes[nn][k]=v
                elif self.poseGraph_closed.nodes[nn].get('modified',False) is True:
                    for k,v in  self.poseGraph_closed.nodes[nn].items():
                        if k in self.node_modified_fields:
                            self.poseGraph.nodes[nn][k]=v
                
            for nn in self.poseGraph.nodes:
                self.poseGraph.nodes[nn]['modified']=False
        
            self.poseGraph_closed = None
        
    def updatePoseGraph(self,msgpkl_sHg_updated):
        print("updating loop closed poses")
        self.poseGraph_closed,self.sHg_updated,self.node_modified_fields,self.edge_modified_fields = pkl.loads(codecs.decode(msgpkl_sHg_updated.data.encode(), "base64"))
        
        # self.mutex.acquire()
        # unpickle the loop closed global frames
        
                    
        # for nn in self.poseGraph.nodes:
        #     if self.poseGraph.nodes[nn].get('frametype',None) is None:
        #         print("no framtype for %d"%nn)
        # print("****FERE***********")
        # for nn in poseGraph_closed.edges:
        #     if nn not in self.poseGraph.edges and poseGraph_closed.edges[nn]["edgetype"]!="Key2Scan":
        #         print("added edge: ",nn,nn[0] in self.poseGraph.nodes,nn[1] in self.poseGraph.nodes)
        #         self.poseGraph.add_edge(nn[0],nn[1])
                
        #         for k,v in  poseGraph_closed.edges[nn].items():
        #             self.poseGraph.edges[nn][k]=v
        #     elif poseGraph_closed.edges[nn].get('modified',False) is True:
        #         if nn in self.poseGraph.edges:
        #             for k,v in  poseGraph_closed.edges[nn].items():
        #                 self.poseGraph.edges[nn][k]=v
        
        # for nn in self.poseGraph.edges:
        #     if self.poseGraph.edges[nn]["edgetype"]!="Key2Scan":
        #         if nn in self.poseGraph.edges:
        #             self.poseGraph.edges[nn]['modified']=False
        
        # # for nn in self.poseGraph.nodes:
        # #     if self.poseGraph.nodes[nn].get('frametype',None) is None:
        # #         print("no framtype for %d"%nn)
        # for nn in poseGraph_closed.nodes:
        #     if nn not in self.poseGraph.nodes and poseGraph_closed.nodes[nn]["frametype"]!="scan":
        #         self.poseGraph.add_node(nn)
        #         print("added new node: ",nn)
        #         for k,v in  poseGraph_closed.nodes[nn].items():
        #             self.poseGraph.nodes[nn][k]=v
        #     elif poseGraph_closed.nodes[nn].get('modified',False) is True:
        #         if nn in self.poseGraph.nodes:
        #             for k,v in  poseGraph_closed.nodes[nn].items():
        #                 self.poseGraph.nodes[nn][k]=v
            
        # for nn in self.poseGraph.nodes:
        #     if self.poseGraph.nodes[nn]["frametype"]!="scan":
        #         if nn in self.poseGraph.nodes:
        #             self.poseGraph.nodes[nn]['modified']=False
            
        
        # update the nodes that went through the process of loop detections
        # Also add the edges
        # mxclosed = max(poseGraph_closed.nodes)
        # mx = max(self.poseGraph.nodes)
        # mn = min(self.poseGraph.nodes)
        
        # L=list(poseGraph_closed.nodes)
        # ng=L[L.index(mxclosed)]
        # LL=[nn for nn in self.poseGraph.nodes if nn>=mxclosed]
        # G=self.poseGraph.subgraph(LL)
        # # poseGraph_closed = nx.compose(poseGraph_closed,G)
        # poseGraph_closed.update(G)
        # for ns in self.poseGraph.nodes:
        #     if ns > mxclosed and ns not in poseGraph_closed.nodes:
        #         poseGraph_closed.add_node(ns)
        #         poseGraph_closed.nodes[ns].update(self.poseGraph.nodes[ns])
        
        # flg=False
        # for (ns1,ns2) in self.poseGraph.edges:
        #     # for ns2 in poseGraph_closed.nodes:
        #     if ns1 in poseGraph_closed.nodes and ns2 in poseGraph_closed.nodes:
        #         if (ns1,ns2) not in poseGraph_closed.edges:
        #             poseGraph_closed.add_edge(ns1,ns2)
        #             poseGraph_closed.edges[ns1,ns2].update(self.poseGraph.edges[ns1,ns2])
        #             flg=True
                    
                    
        # for ns in list(poseGraph_closed.nodes):
        #     for pidx in poseGraph_closed.predecessors(ns):
        #         if poseGraph_closed.nodes[pidx]['frametype']=="keyframe": # and pidx in sHg_updated
        #             if poseGraph_closed.edges[pidx,ns]['edgetype']=="Key2Key" or poseGraph_closed.edges[pidx,ns]['edgetype']=="Key2Scan":
        #                 psHg=poseGraph_closed.nodes[pidx]['sHg']
        #                 nsHps=poseGraph_closed.edges[pidx,ns]['H']
        #                 nsHg = nsHps.dot(psHg)
        #                 poseGraph_closed.nodes[ns]['sHg']=nsHg
        #                 gHns=nplinalg.inv(nsHg)
        #                 tpos=np.matmul(gHns,np.array([0,0,1]))
        #                 poseGraph_closed.nodes[ns]['pos'] = (tpos[0],tpos[1])
            
        #                 break
        
        # self.poseGraph=poseGraph_closed
        

        # with open("April28-2021.pkl",'wb') as fh:
        #     pkl.dump([self.poseGraph,params],fh)
    
        # self.mutex.release()
        
        
#%% TEST::::::: Pose estimation by keyframe


# idx1=1000
# previdx_loopclosure = 0 # 

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtlebotLidarPoseEstimator')
    # node.declare_parameters(
    #     namespace='',
    #     parameters=[
    #         ('my_str', 1),
    #         ('my_int', 2),
    #         ('my_double_array', True)
    #     ]
    # )
    
    # print("param_value is :",node.get_parameter('/turtlebotParameters/my_str').value )
    
    
    pld=ProcessLidarData()
    
    # topic to continuously publish the global pose computed from lidar scans
    lidarpose_pub = node.create_publisher(PoseStamped, 'lidarPose',qos_profile_sensor_data)
    pld.setlidarpose_publisher(lidarpose_pub)
    
    #plot posegraph
    plotpose_pub = node.create_publisher(String, 'posegraphplot',ttlhelp.qos_closedposegraphPoses_profile)
    pld.setplotpose_publisher(plotpose_pub)
    
    # topic to intermittently publish the full posegraph to do loopclosure
    lidarposegraph_pub = node.create_publisher(String, 'posegraphclose',ttlhelp.qos_closedposegraphPoses_profile)
    pld.setlidarposegraph_publisher(lidarposegraph_pub)
        
    # subscribe to scan data
    node.create_subscription(LaserScan,'scan',pld.processScanPts,qos_profile_sensor_data)
    # node.create_subscription(MultiEchoLaserScan,'/horizontal_laser_2d',pld.processScanPts,ttlhelp.qos_scans_profile)
    
    # subscribe to recieve the closed global poses.
    node.create_subscription(String,'posegraphClosedPoses',pld.updatePoseGraph,ttlhelp.qos_closedposegraphPoses_profile)
    
    print("ready")
    try:
        # while True:
            # rclpy.spin_once(node,timeout_sec=0.001)
        rclpy.spin(node)
    except KeyboardInterrupt:
    	pass
    
    # pld.publishPoseGraph()
    
    with open("DeutchesMeuseum.pkl",'wb') as fh:
        pkl.dump([pld.poseGraph,params],fh)
            
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    



    
    
    


#%%
