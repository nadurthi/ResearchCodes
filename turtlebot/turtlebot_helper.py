# -*- coding: utf-8 -*-

import numpy as np
from lidarprocessing import point2Dprocessing as pt2dproc
from sklearn.neighbors import KDTree
import datetime
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Imu, MultiEchoLaserScan

from rclpy.duration import Duration
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSLivelinessPolicy
from rclpy.qos import QoSPresetProfiles
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

qos_closedposegraphPoses_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            lifespan=Duration(seconds=2),
            deadline=Duration(seconds=2),
            # liveliness=QoSLivelinessPolicy.MANUAL_BY_TOPIC,
            # liveliness_lease_duration=Duration(nanoseconds=12),
            avoid_ros_namespace_conventions=True
        )


qos_scans_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=500,
            reliability=QoSReliabilityPolicy.RELIABLE,
            # durability=QoSDurabilityPolicy.VOLATILE,
            # lifespan=Duration(seconds=2),
            # deadline=Duration(seconds=2),
            # liveliness=QoSLivelinessPolicy.MANUAL_BY_TOPIC,
            # liveliness_lease_duration=Duration(nanoseconds=12),
            # avoid_ros_namespace_conventions=True
        )

#%%
# params={}

# params['REL_POS_THRESH']=0.25 # meters after which a keyframe is made
# params['ERR_THRES']=0.7
# params['n_components']=35
# params['reg_covar']=0.002

# params['PLOT_AFTER_#KEYFRAMES'] = 5 # currently not used


# params['doLoopClosure'] = True
# params['LOOP_CLOSURE_D_THES']=1
# params['LOOP_CLOSURE_ERR_THES']= 1.5
# params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.5
# params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = 0.15

# params['LOOP_CLOSURE_POS_THES']=10
# params['LOOP_CLOSURE_POS_MIN_THES']=0.1

# params['LOOPCLOSE_BIN_MATCHER_dx'] = 0.5
# params['LOOPCLOSE_BIN_MATCHER_L'] = 5

# params['LOOPCLOSE_AFTER_#KEYFRAMES'] = 6



# params['Do_GMM_FINE_FIT']=True

# params['Do_BIN_FINE_FIT'] = True

# params['Do_BIN_DEBUG_PLOT-dx']=False
# params['Do_BIN_DEBUG_PLOT']= False

# params['xy_hess_inv_thres']=10000*0.4
# params['th_hess_inv_thres']=100000*0.4
# params['#ThreadsLoopClose']=4

# params['INTER_DISTANCE_BINS_max']=10
# params['INTER_DISTANCE_BINS_dx']=0.15


#%%
# time_increment = 1.736111516947858e-05
# angle_increment = 0.004363323096185923
# scan_time = 0.02500000037252903
# range_min, range_max = 0.023000000044703484, 60.0
# angle_min,angle_max =  -2.3518311977386475,2.3518311977386475


def filter_scanmsg(scanmsg):
    T=datetime.datetime.fromtimestamp(scanmsg.header.stamp.sec+1e-9*scanmsg.header.stamp.nanosec)
    Tstamp = scanmsg.header.stamp
    
    angle_min=scanmsg.angle_min
    angle_max=scanmsg.angle_max
    angle_increment=scanmsg.angle_increment
    range_min = scanmsg.range_min
    range_max = scanmsg.range_max
    
    if isinstance(scanmsg.ranges[0],list):
        rngs = list(map(lambda x: np.max(x) if len(x)>0 else range_max,scanmsg.ranges))    
    
    
    
    
    
    
    if isinstance(scanmsg,MultiEchoLaserScan):
        rngs=[list(scanmsg.ranges[i].echoes) for i in range(len(scanmsg.ranges))]
        rngs = list(map(lambda x: np.max(x) if len(x)>0 else 120,rngs))
        rngs = np.array(rngs)
        
        ths = np.arange(angle_min,angle_max,angle_increment)
        p=np.vstack([np.cos(ths),np.sin(ths)])
        
        rngidx = (rngs> (range_min+0.1) ) & (rngs< (range_max-5))
        ptset = rngs.reshape(-1,1)*p.T
        
    else:
        ths = np.arange(angle_min,angle_max+angle_increment,angle_increment)
        p=np.vstack([np.cos(ths),np.sin(ths)])
        
        rngs = np.array(scanmsg.ranges)
        
        rngidx = (rngs>= scanmsg.range_min) & (rngs<= scanmsg.range_max)
        ptset = rngs.reshape(-1,1)*p.T
        
        
    
    X=ptset[rngidx,:]
    
    Xd=pt2dproc.binnerDownSampler(X,dx=0.025,cntThres=1)
                
    # now filter silly points
    tree = KDTree(Xd, leaf_size=5)
    cnt = tree.query_radius(Xd, 0.25,count_only=True) 
    Xd= Xd[cnt>=2,:]
    
    cnt = tree.query_radius(Xd, 0.5,count_only=True) 
    Xd = Xd[cnt>=5,:]
    
    return Tstamp,T,Xd


