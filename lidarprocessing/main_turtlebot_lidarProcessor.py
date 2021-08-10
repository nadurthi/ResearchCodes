# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:46:31 2020

@author: nadur
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data, qos_profile_system_default
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Imu 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose,PoseStamped


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

qos_closedposegraphPoses_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            lifespan=Duration(seconds=4),
            deadline=Duration(seconds=5),
            # liveliness=QoSLivelinessPolicy.MANUAL_BY_TOPIC,
            # liveliness_lease_duration=Duration(nanoseconds=12),
            avoid_ros_namespace_conventions=True
        )


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
    
    
REL_POS_THRESH=0.3 # meters after which a keyframe is made
ERR_THRES=1.2 # error threshold after which a keyframe is made


LOOP_CLOSURE_D_THES=1.5 # the histogram threshold for matching
LOOP_CLOSURE_POS_THES=4 # match two keyframes only if they are within this threshold
LOOP_CLOSURE_ERR_THES=-0.70 # match two keyframes only if error is less than this threshold


class ProcessLidarData:
    def __init__(self):
        self.poseGraph = nx.DiGraph()
        self.idx0 = 0
        self.loopClosedFrameidx = 0
        
    def setlidarpose_publisher(self,lidarpose_pub):
        self.lidarpose_pub = lidarpose_pub
        
    def setlidarposegraph_publisher(self,lidarposegraph_pub):
        self.lidarposegraph_pub = lidarposegraph_pub
    
    def publishPoseGraph(self):
        msg=String()
        # we need to use this weird codecs to encode non-ASCI characters in the pickled string
        # ros does not accept non-ASCI characters. 
        pickled = codecs.encode(pkl.dumps(self.poseGraph), "base64").decode()
        msg.data = pickled
        self.lidarposegraph_pub.publish(msg)
        
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
        # ranges = dataset[i]['ranges']
        T=datetime.datetime.fromtimestamp(scanmsg.header.stamp.sec+1e-9*scanmsg.header.stamp.nanosec)
        Tstamp = scanmsg.header.stamp
        
        rngs = np.array(scanmsg.ranges)
        
        angle_min=scanmsg.angle_min
        angle_max=scanmsg.angle_max
        angle_increment=scanmsg.angle_increment
        ths = np.arange(angle_min,angle_max+angle_increment,angle_increment)
        p=np.vstack([np.cos(ths),np.sin(ths)])
        
        rngidx = (rngs>= scanmsg.range_min) & (rngs<= scanmsg.range_max)
        ptset = rngs.reshape(-1,1)*p.T
        
        X=ptset[rngidx,:]
        
        self.computePose(Tstamp,T,X)
        # return X


    def computePose(self,Tstamp,T,X):
        # first frame is automatically a keyframe
        if len(self.poseGraph.nodes)==0:
            Xd,m = pt2dproc.get0meanIcov(X)
            clf,MU,P,W = pt2dproc.getclf(Xd)
            H=np.hstack([np.identity(2),np.zeros((2,1))])
            H=np.vstack([H,[0,0,1]])
            h=pt2dproc.get2DptFeat(X)
            self.idx=self.idx0
            # saving the data X can be removed later when we revise the code. Saving X might consume time
            self.poseGraph.add_node(self.idx,frametype="keyframe",clf=clf,X=X,m_clf=m,time=T,sHg=H,pos=(0,0),h=h,color='g',LoopDetectDone=False)
            self.KeyFrame_prevIdx=self.idx
            
            self.idx+=1
            
            return True

        # estimate pose to last keyframe
        KeyFrameClf = self.poseGraph.nodes[self.KeyFrame_prevIdx]['clf']
        m_clf = self.poseGraph.nodes[self.KeyFrame_prevIdx]['m_clf']
        
        # get a initial guess for pose optimization using prev frame pose. 
        if (self.idx-self.KeyFrame_prevIdx)<=1: # if it is too close use Idenity as initial guess of pose
            sHk_prevframe = np.identity(3)
        else: # or else get the H pose matrix between prev frame and prev scan frame
            sHk_prevframe = self.poseGraph.edges[self.KeyFrame_prevIdx,self.idx-1]['H']

        # assuming sHk_prevframe is very close to sHk, use it as a guess for pose optimization
        # now match the point cloud X to the previous keyframe
        st=time.time()
        sHk,err = pt2dproc.scan2keyframe_match(KeyFrameClf,m_clf,X,sHk=sHk_prevframe)
        et = time.time()
                
        print("idx = ",self.idx," Error = ",err," , and time taken = ",et-st)
        # now get the global pose to the frame
        kHg = self.poseGraph.nodes[self.KeyFrame_prevIdx]['sHg'] #global pose to the prev keyframe
        sHg = np.matmul(sHk,kHg) # global pose to the current frame: global to current frame
        gHs=nplinalg.inv(sHg) # current frame to global


        # check if you have to make this the keyframe
        # also if the frame id more than 100 frame away, make it a keyframe
        # when you make it a keyframe, compute the gmm classfier for it to be used in subsequent matchings
        if err>ERR_THRES or nplinalg.norm(sHk[:2,2])>REL_POS_THRESH or (self.idx-self.KeyFrame_prevIdx)>100: 
            print("New Keyframe will now be added")
            Xd,m = pt2dproc.get0meanIcov(X)
            clf,MU,P,W = pt2dproc.getclf(Xd)
            tpos=np.matmul(gHs,np.array([0,0,1])) 
    
            h=pt2dproc.get2DptFeat(X)
            # saving the data X can be removed later when we revise the code. Saving X might consume time
            self.poseGraph.add_node(self.idx,frametype="keyframe",clf=clf,X=X,m_clf=m,time=T,sHg=sHg,pos=(tpos[0],tpos[1]),h=h,color='g',LoopDetectDone=False)
            # Now add an edge from prev keyframe to current keyframe, call it Key2Key edge
            self.poseGraph.add_edge(self.KeyFrame_prevIdx,self.idx,H=sHk,edgetype="Key2Key",color='k')
            
            # now delete previous scan data up-until the previous keyframe
            # this is to save space. but keep 1 the mid scan frame as it might be useful later
            #Also complete pose estimation to this scan from the new keyframe, again just in case it might be useful
            pt2dproc.addedge2midscan(self.poseGraph,self.idx,self.KeyFrame_prevIdx,sHk,keepOtherScans=False)
                
            # make the current idx as the previous keyframe 
            self.KeyFrame_prevIdx = self.idx

                            
        else: #not a keyframe
            # add the scan frame 
            tpos=np.matmul(gHs,np.array([0,0,1]))
            # saving the data X can be removed later when we revise the code. Saving X might consume time
            self.poseGraph.add_node(self.idx,frametype="scan",time=T,X=X,sHg=sHg,pos=(tpos[0],tpos[1]),color='r',LoopDetectDone=False)
            self.poseGraph.add_edge(self.KeyFrame_prevIdx,self.idx,H=sHk,edgetype="Key2Scan",color='r')
        
        self.idx+=1
        self.publishPose(Tstamp,gHs)
        
        # request a loop closure after every say 50 frames
        # when we publish the posegraph on topic "posegraphclose", 
        # the other node with do the loop closure and send it back on topic  "lidarLoopClosedPoses" with callback "updatePoseGraph" below
        if self.idx-self.loopClosedFrameidx>50:
            self.loopClosedFrameidx = self.idx
            print("publish posegraph to call for loop closure")
            self.publishPoseGraph()
             # with lots of frames saving the points X, the posegrph will be large
             # for now let it be, but later we can think of saving the scan points in a separate node/server
        
    def updatePoseGraph(self,msgpkl_sHg_updated):
        print("updating loop closed poses")
        # unpickle the loop closed global frames
        sHg_updated,LoopDetectionsDoneDict = pkl.loads(codecs.decode(msgpkl_sHg_updated.data.encode(), "base64"))
       
        
        self.poseGraph=pt2dproc.updateGlobalPoses(self.poseGraph,sHg_updated)
        
        # update the nodes that went through the process of loop detections
        # this way, next time we call for loop closure, we can avoid these nodes
        for ns in LoopDetectionsDoneDict:
            if ns in self.poseGraph.nodes:
                self.poseGraph.nodes[ns]['LoopDetectDone'] = LoopDetectionsDoneDict[ns]
#%% TEST::::::: Pose estimation by keyframe


# idx1=1000
# previdx_loopclosure = 0 # 

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtlebotLidarPoseEstimator')
    
    pld=ProcessLidarData()
    
    # topic to continuously publish the global pose computed from lidar scans
    lidarpose_pub = node.create_publisher(PoseStamped, 'lidarPose',qos_profile_sensor_data)
    pld.setlidarpose_publisher(lidarpose_pub)
    
    # topic to intermittently publish the full posegraph to do loopclosure
    lidarposegraph_pub = node.create_publisher(String, 'posegraphclose',qos_profile_sensor_data)
    pld.setlidarposegraph_publisher(lidarposegraph_pub)
        
    # subscribe to scan data
    node.create_subscription(LaserScan,'scan',pld.processScanPts,qos_profile_sensor_data)
    
    # subscribe to recieve the closed global poses.
    node.create_subscription(String,'posegraphClosedPoses',pld.updatePoseGraph,qos_closedposegraphPoses_profile)
    
    print("ready")
    try:
        # while True:
            # rclpy.spin_once(node,timeout_sec=0.001)
        rclpy.spin(node)
    except KeyboardInterrupt:
    	pass
    
    pld.publishPoseGraph()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    



    
    
    


#%%
