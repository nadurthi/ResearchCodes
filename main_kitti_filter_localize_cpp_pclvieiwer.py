#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 22:01:28 2021

@author: nagnanmus
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
def quat2omega_scipyspline(tvec,qvec,k=2,s=0.001):
    spl0 = UnivariateSpline(tvec, qvec[:,0], k=k,s=s)
    spl1 = UnivariateSpline(tvec, qvec[:,1], k=k,s=s)
    spl2 = UnivariateSpline(tvec, qvec[:,2], k=k,s=s)
    spl3 = UnivariateSpline(tvec, qvec[:,3], k=k,s=s)
    
    spld0=spl0.derivative()
    spld1=spl1.derivative()
    spld2=spl2.derivative()
    spld3=spl3.derivative()
    
    qdot = np.zeros_like(qvec)
    qvec_sp = np.zeros_like(qvec)
    
    qvec_sp[:,0] = spl0(tvec)
    qvec_sp[:,1] = spl1(tvec)
    qvec_sp[:,2] = spl2(tvec)
    qvec_sp[:,3] = spl3(tvec)
    
    qdot[:,0] = spld0(tvec)
    qdot[:,1] = spld1(tvec)
    qdot[:,2] = spld2(tvec)
    qdot[:,3] = spld3(tvec)
    
    qqdot=quaternion.from_float_array(qdot)
    qqvec=quaternion.from_float_array(qvec_sp)
    
    w=np.zeros((len(qvec),4))
    for i in range(len(qvec)):
        pp=quaternion.as_float_array(qqdot[i]*qqvec[i].inverse())
        w[i]= 2*pp
    
    spl0 = UnivariateSpline(tvec, w[:,0], k=k,s=s)
    spl1 = UnivariateSpline(tvec, w[:,1], k=k,s=s)
    spl2 = UnivariateSpline(tvec, w[:,2], k=k,s=s)
    spl3 = UnivariateSpline(tvec, w[:,3], k=k,s=s)
    
    spld0=spl0.derivative()
    spld1=spl1.derivative()
    spld2=spl2.derivative()
    spld3=spl3.derivative()
    
    alpha = np.zeros((len(qvec),4))
    alpha[:,0] = spld0(tvec)
    alpha[:,1] = spld1(tvec)
    alpha[:,2] = spld2(tvec)
    alpha[:,3] = spld3(tvec)
    
    return tvec,qvec_sp,w,qdot,alpha
#%%
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

try:
    runfilename = __file__
except:
    runfilename = "/home/na0043/Insync/n.adurthi@gmail.com/Google Drive/repos/SLAM/main_kitti_filter_localize_cpp.py"
    
metalog="""
Journal paper KITTI localization using Particle filter
Author: Venkat
Date: Feb 22 2022

"""
dt=dataset.times[1]-dataset.times[0]
simmanger = uqsimmanager.SimManager(t0=dataset.times[0],tf=dataset.times[-1],dt=dt,dtplot=dt/10,
                                  simname="KITTI-localization-%s"%(sequence),savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()



nd=len(dataset.poses)

Xtpath=np.zeros((len(dataset),10))

for i in range(len(dataset)):
    H=dataset.calib.T_cam0_velo
    H=np.dot(nplinalg.inv(H),dataset.poses[i].dot(H))
    
    Xtpath[i,0:3] = H.dot(np.array([0,0,0,1]))[0:3]
    # Xtpath[i,0:3] = dataset.poses[i].dot(np.array([0,0,0,1]))[0:3]
    
    r = Rsc.from_matrix(H[0:3,0:3])
    
    q =r.as_euler('zyx',degrees=False)
    Xtpath[i,3:6] = q
    
    q =r.as_quat()
    Xtpath[i,6:] = q

tvec,qvec_sp,w,qdot,alpha=quat2omega_scipyspline(dataset.times,Xtpath[:,6:],k=3,s=0.0001)


    
Velocities=np.zeros((len(dataset),3))
Acc=np.zeros((len(dataset),3))

splx=UnivariateSpline(dataset.times,Xtpath[:,0])
splvx=splx.derivative()
splax=splvx.derivative()

sply=UnivariateSpline(dataset.times,Xtpath[:,1])
splvy=sply.derivative()
splay=splvy.derivative()

splz=UnivariateSpline(dataset.times,Xtpath[:,2])
splvz=splz.derivative()
splaz=splvz.derivative()


Velocities[:,0]=splvx(dataset.times)
Velocities[:,1]=splvy(dataset.times)
Velocities[:,2]=splvz(dataset.times)

Acc[:,0]=splax(dataset.times)
Acc[:,1]=splay(dataset.times)
Acc[:,2]=splaz(dataset.times)

rotations = Rsc.from_euler('zyx', Xtpath[:,3:6], degrees=False)
spline =RscSpl(dataset.times, rotations)
AngRates=spline(dataset.times, 1)
AngAcc=spline(dataset.times, 2)
AngRates=AngRates[:,::-1]
AngAcc=AngAcc[:,::-1]

# AngRates=w[:,1:]
# AngAcc=alpha[:,1:]

# yawspl=UnivariateSpline(dataset.times,Xtpath[:,3],k=5)
# pitchspl=UnivariateSpline(dataset.times,Xtpath[:,4],k=5)
# rollspl=UnivariateSpline(dataset.times,Xtpath[:,5],k=5)

# yawrate=yawspl.derivative()(dataset.times)
# pitchrate=pitchspl.derivative()(dataset.times)
# rollrate=rollspl.derivative()(dataset.times)

# AngRates=np.vstack([yawrate,pitchrate,rollrate]).T
# AngAcc=alpha[:,1:]



def getmeas(k):
    Hgt=dataset.calib.T_cam0_velo
    Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[k].dot(Hgt))
    X1v = dataset.get_velo(k)
    X1v=X1v[:,:3]
    
    idxx=(X1v[:,0]>-200) & (X1v[:,0]<200) & (X1v[:,1]>-200) & (X1v[:,1]<200 )& (X1v[:,2]>-100) & (X1v[:,2]<100)
    X1v=X1v[idxx,:]
    
    X1gv=Hgt[0:3,0:3].dot(X1v.T).T+Hgt[0:3,3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X1v[:,:3])
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    X1vnorm = np.asarray(downpcd.normals)
    idd=np.abs(X1vnorm[:,2])<0.5
    X1v_roadrem=np.asarray(downpcd.points)
    X1v_roadrem=X1v_roadrem[idd] 
    X1gv_roadrem=Hgt[0:3,0:3].dot(X1v_roadrem.T).T+Hgt[0:3,3]
    if k==0:
        dt=0
    else:
        dt=dataset.times[k]-dataset.times[k-1]
    tk=dataset.times[k]
    
    return dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem

def getmeas_Q(outQ):
    for k in range(len(dataset.times)):
        outQ.put(getmeas(k))
        print("putted k = ",k)
        
# if __name__=="__main__":
#     outQ = mp.Queue(15)
#     p = mp.Process(target=getmeas_Q, args=(outQ,))
#     p.start()


plt.close("all")
#%%

D={}


D["Localize"]={}
D["Localize"]["sig0"]=0.25
D["Localize"]["dmax"]=10
D["Localize"]["likelihoodsearchmethod"]="octree" # lookup
D["Localize"]["octree"]={"resolution":0.1,
                         "searchmethod":"ApprxNN"}
D["Localize"]["lookup"]={"resolution":[0.25,0.25,0.25]}



#
D["MapManager"]={"map":{},"map2D":{}}
D["MapManager"]["map"]["downsample"]={"enable":True,"resolution":[0.2,0.2,0.2]}
D["MapManager"]["map2D"]["downsample"]={"enable":True,"resolution":[0.1,0.1,0.1]}
D["MapManager"]["map2D"]["removeRoad"]=False

#
D["MeasMenaager"]={"meas":{},"meas2D":{},"meas_Likelihood":{}}
D["MeasMenaager"]["meas"]["downsample"]={"enable":True,"resolution":[0.2,0.2,0.2]}
D["MeasMenaager"]["meas_Likelihood"]["downsample"]={"enable":True,"resolution":[0.3,0.3,0.3]}
D["MeasMenaager"]["meas2D"]["removeRoad"]=False
D["MeasMenaager"]["meas2D"]["downsample"]={"enable":True,"resolution":[0.1,0.1,0.1]}


#
D["BinMatch"]={}
D['BinMatch']["dxMatch"]=list(np.array([1,1],dtype=np.float64))
D['BinMatch']["dxBase"]=list(np.array([30,30],dtype=np.float64))
D['BinMatch']["Lmax"]=list(np.array([200,200],dtype=np.float64))
D['BinMatch']["thmax"]=170*np.pi/180
D['BinMatch']["thfineres"]=2.5*np.pi/180

#
D["mapfit"]={"downsample":{},"gicp":{}}
D["mapfit"]["downsample"]={"resolution":[0.5,0.5,0.5]}
D["mapfit"]["gicp"]["setMaxCorrespondenceDistance"]=10
D["mapfit"]["gicp"]["setMaximumIterations"]=5.0
D["mapfit"]["gicp"]["setMaximumOptimizerIterations"]=5.0
D["mapfit"]["gicp"]["setRANSACIterations"]=0
D["mapfit"]["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["mapfit"]["gicp"]["setTransformationEpsilon"]=1e-6
D["mapfit"]["gicp"]["setUseReciprocalCorrespondences"]=1

#
D["seqfit"]={"downsample":{},"gicp":{}}
D["seqfit"]["gicp"]["setMaxCorrespondenceDistance"]=10
D["seqfit"]["gicp"]["setMaximumIterations"]=30.0
D["seqfit"]["gicp"]["setMaximumOptimizerIterations"]=30.0
D["seqfit"]["gicp"]["setRANSACIterations"]=0
D["seqfit"]["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["seqfit"]["gicp"]["setTransformationEpsilon"]=1e-9
D["seqfit"]["gicp"]["setUseReciprocalCorrespondences"]=1


D["plotting"]={}
D["plotting"]["map_color"]=[211,211,211]
D["plotting"]["map_pointsize"]=2
D["plotting"]["traj_color"]=[0,0,211]
D["plotting"]["traj_pointsize"]=2
D["plotting"]["pf_color"]=[211,0,0]
D["plotting"]["pf_pointsize"]=2
D["plotting"]["pf_arrowlen"]=5
#%% PF 


pcd3D=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)
pcd3Droadremove=o3d.io.read_point_cloud("kitti-pcd-seq-roadremove-%s.pcd"%sequence)

pcd3DdownSensorCost=pcd3D.voxel_down_sample(voxel_size=0.5)

Xmap =np.asarray(pcd3D.points)
Xmap2D=np.asarray(pcd3Droadremove.points)


pcd2ddown = o3d.geometry.PointCloud()
Xmap2Dflat=Xmap2D.copy()
Xmap2Dflat[:,2]=0
pcd2ddown.points = o3d.utility.Vector3dVector(Xmap2Dflat[:,:3])
pcd2ddown = pcd2ddown.voxel_down_sample(voxel_size=1)

Xmap2Dflat=np.asarray(pcd2ddown.points)

#%%

KL=kittilocal.MapLocalizer(json.dumps(D))
KL.addMap(Xmap)

X=np.random.randn(15,10)
KL.plotsim(X)
