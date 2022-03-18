#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:25:11 2022

@author: na0043
"""
import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from threading import Thread
import os
from utils.plotting import geometryshapes as utpltgmshp   
import pandas as pd
from utils import simmanager as uqsimmanager
from lidarprocessing.numba_codes import gicp
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import RotationSpline as RscSpl 
dtype = np.float64
import time
import open3d as o3d
from pykitticustom import odometry
from  uq.filters import pf 
import multiprocessing as mp
import queue

from scipy.interpolate import UnivariateSpline

import quaternion
import pickle
from joblib import dump, load

from pyslam import  slam
from pyslam import kittilocal

import json

plt.close("all")
# basedir =r'P:\SLAMData\Kitti\visualodo\dataset'

basedir ='/media/na0043/misc/DATA/KITTI/odometry/dataset'
# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
loop_closed_seq = ['02','05','06','08']
sequence = '05'





dataset = odometry.odometry(basedir, sequence, frames=None) # frames=range(0, 20, 5)

pcd3D=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)



XX=np.asarray(pcd3D.points)
bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(XX,axis=0),max_bound=np.max(XX,axis=0)-150)
# bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([0,0]),max_bound=np.array([150,150]))

Xv=dataset.get_velo(0)
pcdX1gv_pose = o3d.geometry.PointCloud()
pcdX1gv_pose.points = o3d.utility.Vector3dVector(Xv[:,:3])
pcdX1gv_pose.paint_uniform_color([0,1,0]) #green

pcdbox=pcd3D.crop(bbox3d)
pcdbox.paint_uniform_color([211/255,211/255,211/255])

o3d.visualization.draw_geometries([pcd3D]) #line_set

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcdX1gv_pose)
ctr = vis.get_view_control()
print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
ctr.change_field_of_view(step=90)
print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
vis.run()
vis.destroy_window()

