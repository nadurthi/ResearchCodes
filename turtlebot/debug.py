# -*- coding: utf-8 -*-


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




with open("turtlebot/DeutchesMeuseum.pkl",'rb') as fh:
    poseGraph,params=pkl.load(fh)
    

Lkeyloop = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))

pt2dplot.plot_keyscan_path(poseGraph,Lkeyloop[0],Lkeyloop[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=True,
                                   forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)



Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))
# Ledges = poseGraph.edges

for previdx,idx  in Ledges:
    if previdx>=5102 and previdx<=5229:
        pass
    else:
        continue
    # posematch = pt2dproc.poseGraph_keyFrame_matcher_binmatch(poseGraph,previdx,idx,params,DoCLFmatch=True,dx0=0.8,L0=1,th0=np.pi/12,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
    
    # posematch=poseGraph.edges[previdx,idx]['posematch']
    mbinfrac=posematch['mbinfrac']
    mbinfrac_ActiveOvrlp=posematch['mbinfrac_ActiveOvrlp']
    
    piHi=posematch['H']
    
    # piHi=poseGraph.edges[previdx,idx]['H']
    
    pt2dplot.plotcomparisons(poseGraph,previdx,idx,UseLC=False,H12=nplinalg.inv(piHi),err=mbinfrac_ActiveOvrlp) #nplinalg.inv(piHi) 
    fig = plt.figure("ComparisonPlot")
    fig.savefig("Key2Key-%d-%d.png"%(idx, previdx))
    plt.close(fig)
    

