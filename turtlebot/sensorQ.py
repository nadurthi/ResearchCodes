#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:03:42 2021

@author: na0043

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

import multiprocessing as mp
import threading
import queue


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
class ProcessScans:
    def __init__(self):
        self.scans=[]
        
        self.pub=None
    def readScans(self,scanmsg):
        self.scans.append(scanmsg)
        
    def sendScans(self):
        if len(self.scans)==0:
            return
        scanmsg=self.scans.pop(0)
        
        self.pub.publish(scanmsg)
        print("Q-len = ",len(self.scans))
        
# idx1=1000
# previdx_loopclosure = 0 # 

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('SensorScanQ')
    # node.declare_parameters(
    #     namespace='',
    #     parameters=[
    #         ('my_str', 1),
    #         ('my_int', 2),
    #         ('my_double_array', True)
    #     ]
    # )
    
    # print("param_value is :",node.get_parameter('/turtlebotParameters/my_str').value )
    
    
    pld=ProcessScans()
    
    node.create_subscription(MultiEchoLaserScan,'/horizontal_laser_2d',pld.readScans)
    
    # topic to continuously publish the global pose computed from lidar scans
    pub = node.create_publisher(MultiEchoLaserScan, 'scan1',ttlhelp.qos_scans_profile)
    pld.pub=pub
    node.create_timer(0.1,pld.sendScans)
   
    print("ready")
    try:
        # while True:
            # rclpy.spin_once(node,timeout_sec=0.001)
            # pld.sendScans()
        rclpy.spin(node)
    except KeyboardInterrupt:
    	pass
    
            
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    


