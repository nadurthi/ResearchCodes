# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:46:31 2020

@author: nadur
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Imu 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose


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
# from sklearn import mixture
# from sklearn.neighbors import KDTree
# from uq.gmm import gmmfuncs as uqgmmfnc
# from utils.plotting import geometryshapes as utpltgmshp
import time
# from scipy.optimize import minimize, rosen, rosen_der,least_squares
# from scipy import interpolate
import networkx as nx
# import pdb
# import pandas as pd
# from fastdist import fastdist
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


class ProcessLoopClose:
    def __init__(self):
        pass
    
    def setlidarloopclose_publisher(self,loopclosedposes_pub):
        self.loopclosedposes_pub=loopclosedposes_pub
        
    def getPoseGraph(self,msg):
        unpickled = pkl.loads(codecs.decode(msg.data.encode(), "base64"))
        self.poseGraph = unpickled
        self.optimizeLoopClosures()
    
    def optimizeLoopClosures(self):            
        # do the optimization to adjust global poses
        
        Lkeys = list(filter(lambda x: self.poseGraph.nodes[x]['frametype']=="keyframe",self.poseGraph.nodes))
        
        if len(Lkeys)>2:
            print("doing loop closure")
            self.poseGraph=pt2dproc.detectAllLoopClosures(self.poseGraph,LOOP_CLOSURE_D_THES,LOOP_CLOSURE_POS_THES,LOOP_CLOSURE_ERR_THES,returnCopy=False)
            idx0=min(Lkeys)
            idx1=max(Lkeys)

            res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(self.poseGraph,idx0,idx1)
            if res.success:
                msg = String()
                LoopDetectionsDoneDict = nx.get_node_attributes(self.poseGraph,'LoopDetectDone')
                pickled = codecs.encode(pkl.dumps([sHg_updated,LoopDetectionsDoneDict]), "base64").decode()
                msg.data = pickled
                self.loopclosedposes_pub.publish(msg)
                print("sending loop closed poses")
        else:
            print("not enought keyframes")
    
#%% TEST::::::: Pose estimation by keyframe


# idx1=1000
# previdx_loopclosure = 0 # 

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtlebotLoopCloser')
    
    plc=ProcessLoopClose()
    
    loopclosedposes_pub = node.create_publisher(String,'posegraphClosedPoses',qos_closedposegraphPoses_profile)
    plc.setlidarloopclose_publisher(loopclosedposes_pub)

    node.create_subscription(String,'posegraphclose',plc.getPoseGraph,qos_profile_sensor_data)
    
    print("ready")
    try:
        # while True:
            # rclpy.spin_once(node,timeout_sec=0.001)
        rclpy.spin(node)
    except KeyboardInterrupt:
    	pass
    

    rclpy.shutdown()


if __name__ == '__main__':
    main()
    



    
    
    


#%%
