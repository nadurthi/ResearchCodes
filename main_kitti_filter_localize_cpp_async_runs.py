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
import sys
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from threading import Thread
import os
import copy
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
import yaml
from scipy.interpolate import UnivariateSpline

import quaternion
import pickle
from joblib import dump, load

from pyslam import  slam
from pyslam import kittilocal
import argparse
from sklearn.cluster import KMeans
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
# basedir  = '/media/na0043/misc/DATA/KITTI/odometry/dataset'
# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
# loop_closed_seq = ['02','05','06','08']
# sequence = '05'

# D={}

# D['Data']={'folder':'/media/na0043/misc/DATA/KITTI/odometry/dataset','sequence':'05'}

# D['PF']={'Npf':1000,'k0':0}

# D["Localize"]={}
# D["Localize"]["sig0"]=0.45
# D["Localize"]["dmax"]=10
# D["Localize"]["likelihoodsearchmethod"]="octree" # lookup
# D["Localize"]["octree"]={"resolution":0.1,
#                           "searchmethod":"ApprxNN"}
# D["Localize"]["lookup"]={"resolution":[0.25,0.25,0.25]}



# #
# D["MapManager"]={"map":{},"map2D":{},"mapnormals":{}}

# D["MapManager"]["map"]["downsample"]={"enable":True,"resolution":[0.2,0.2,0.2]}
# D["MapManager"]["mapnormals"]={"enable":True,"resolution":[0.05,0.05,0.05],"radius":1,"min_norm_z_thresh":0.5,"knn":15}
# D["MapManager"]["map2D"]["downsample"]={"enable":True,"resolution":[0.1,0.1,0.1]}
# D["MapManager"]["map2D"]["removeRoad"]=False

# #
# D["MeasMenaager"]={"meas":{},"meas2D":{},"meas_Likelihood":{}}
# D["MeasMenaager"]["meas"]["downsample"]={"enable":True,"resolution":[0.2,0.2,0.2]}
# D["MeasMenaager"]["measnormals"]={"enable":True,"resolution":[0.05,0.05,0.05],"radius":1,"min_norm_z_thresh":0.5,"knn":15}
# D["MeasMenaager"]["meas_Likelihood"]["downsample"]={"enable":True,"resolution":[1,1,1]}
# D["MeasMenaager"]["meas2D"]["removeRoad"]=False
# D["MeasMenaager"]["meas2D"]["downsample"]={"enable":True,"resolution":[0.1,0.1,0.1]}


# #
# D["BinMatch"]={}
# D['BinMatch']["dxMatch"]=[1.0,1.0]
# D['BinMatch']["dxBase"]=[30.0,30.0]
# D['BinMatch']["Lmax"]=[200.0,200.0]
# D['BinMatch']["thmax"]=170*np.pi/180
# D['BinMatch']["thfineres"]=2.5*np.pi/180

# #
# D["mapfit"]={"downsample":{},"gicp":{}}
# D["mapfit"]["downsample"]={"resolution":[0.5,0.5,0.5]}
# D["mapfit"]["gicp"]["setMaxCorrespondenceDistance"]=10
# D["mapfit"]["gicp"]["setMaximumIterations"]=5
# D["mapfit"]["gicp"]["setMaximumOptimizerIterations"]=10
# D["mapfit"]["gicp"]["setRANSACIterations"]=0
# D["mapfit"]["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
# D["mapfit"]["gicp"]["setTransformationEpsilon"]=1e-6
# D["mapfit"]["gicp"]["setUseReciprocalCorrespondences"]=1

# #
# D["seqfit"]={"downsample":{},"gicp":{}}
# D["seqfit"]["downsample"]={"resolution":[0.5,0.5,0.5]}
# D["seqfit"]["gicp"]["setMaxCorrespondenceDistance"]=10
# D["seqfit"]["gicp"]["setMaximumIterations"]=5.0
# D["seqfit"]["gicp"]["setMaximumOptimizerIterations"]=10.0
# D["seqfit"]["gicp"]["setRANSACIterations"]=0
# D["seqfit"]["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
# D["seqfit"]["gicp"]["setTransformationEpsilon"]=1e-9
# D["seqfit"]["gicp"]["setUseReciprocalCorrespondences"]=1


# D["plotting"]={}
# D["plotting"]["map_color"]=[211,211,211]
# D["plotting"]["map_pointsize"]=2
# D["plotting"]["traj_color"]=[0,0,211]
# D["plotting"]["traj_pointsize"]=2
# D["plotting"]["pf_color"]=[211,0,0]
# D["plotting"]["pf_pointsize"]=2
# D["plotting"]["pf_arrowlen"]=5


# with open('kitti_localize_config.yml', 'w') as outfile:
#     yaml.dump(D, outfile, default_flow_style=False)

with open('kitti_localize_config.yml', 'r') as outfile:
    D=yaml.safe_load( outfile)

#%%

if __name__=="__main__":
    plt.close("all")
    # basedir =r'P:\SLAMData\Kitti\visualodo\dataset'
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--k0", help="start time step k0",type=int)
    args = parser.parse_args()
    
    
    
    
    
    
    
    dataset = odometry.odometry(D['Data']['folder'], D['Data']['sequence'], frames=None) # frames=range(0, 20, 5)
    
    try:
        runfilename = __file__
    except:
        runfilename = "/home/na0043/Insync/n.adurthi@gmail.com/Google Drive/repos/SLAM/main_kitti_filter_localize_cpp_async_runs.py"
        
    metalog="""
    Journal paper KITTI localization using Particle filter
    Author: Venkat
    Date: Feb 22 2022
    
    """
    dt=dataset.times[1]-dataset.times[0]
    simmanger = uqsimmanager.SimManager(t0=dataset.times[0],tf=dataset.times[-1],dt=dt,dtplot=dt/10,
                                      simname="KITTI-localization-%s"%(D['Data']['sequence']),savepath="simulations",
                                      workdir=os.getcwd())
    
    simmanger.initialize()
    
    simmanger.data['sequence']=D['Data']['sequence']
    simmanger.data['basedir']=D['Data']['folder']
    simmanger.data['D']=copy.deepcopy(D)
    
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
    
    sequence=D['Data']['sequence']   
    
    
    Npf = simmanger.data['Npf']=D['PF']['Npf']
    if args.k0 is None:
        k0 = simmanger.data['k0'] = D['PF']['k0']
    else:
        k0 = simmanger.data['k0'] = args.k0
        print("k0 is set as = ",k0)
        
    KL=kittilocal.MapLocalizer(json.dumps(D))
    #  /media/na0043/misc/DATA/KITTI/odometry/dataset/sequences/00/velodyne
    velofolder = os.path.join(D['Data']['folder'],"sequences",D['Data']['sequence'],"velodyne")
    KL.autoReadMeas_async(velofolder,k0)
    
    # X=KL.getMeasQ_eigen(False)
    # X1v = dataset.get_velo(0)
    
    # pcdX1v = o3d.geometry.PointCloud()
    # pcdX1v.points = o3d.utility.Vector3dVector(X1v[:,:3])
    # pcdX1v.paint_uniform_color([1,0,0]) #green
    
    # pcdX0 = o3d.geometry.PointCloud()
    # pcdX0.points = o3d.utility.Vector3dVector(X[0])
    # pcdX0.paint_uniform_color([0,0,1]) #green
    
    # o3d.visualization.draw_geometries([pcdX1v,pcdX0])
    
    
    
    # pcdX1 = o3d.geometry.PointCloud()
    # pcdX1.points = o3d.utility.Vector3dVector(X[1])
    # pcdX1.paint_uniform_color([0,1,0]) #green
    
    # o3d.visualization.draw_geometries([pcdX0,pcdX1])
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
    
    
    
    KL.addMap(Xmap)
    KL.addMap2D(Xmap2D[:,:2])
    KL.setLookUpDist("kitti-pcd-lookupdist-seq-%s.bin"%sequence)
    #%%
    
    # KL.resetsim()
    
    time.sleep(2)
    
    minx,miny,minz,maxx,maxy,maxz=KL.MapPcllimits()
    
    
    
    
    
    
    
    plt.close("all")
    
    
    
    
    Xlimits_scan=[minx,maxx]
    Ylimits_scan=[miny,maxy]
    Zlimits_scan=[minz,maxz]
    
    
    Xlimits=[np.min(Xtpath[:,0]),np.max(Xtpath[:,0])]
    Ylimits=[np.min(Xtpath[:,1]),np.max(Xtpath[:,1])]
    Zlimits=[np.min(Xtpath[:,2]),np.max(Xtpath[:,2])]
    
    yawlimits=[np.min(Xtpath[:,3]),np.max(Xtpath[:,3])]
    pitchlimits=[np.min(Xtpath[:,4]),np.max(Xtpath[:,4])]
    rolllimits=[np.min(Xtpath[:,5]),np.max(Xtpath[:,5])]
    
    vv=nplinalg.norm(Velocities,axis=1)
    vlimits=[0,np.max(vv)]
    
    Vxlimits=[np.min(Velocities[:,0]),np.max(Velocities[:,0])]
    Vylimits=[np.min(Velocities[:,1]),np.max(Velocities[:,1])]
    Vzlimits=[np.min(Velocities[:,2]),np.max(Velocities[:,2])]
    
    Axlimits=[np.min(Acc[:,0]),np.max(Acc[:,0])]
    Aylimits=[np.min(Acc[:,1]),np.max(Acc[:,1])]
    Azlimits=[np.min(Acc[:,2]),np.max(Acc[:,2])]
    
    omgyawlimits=[np.min(AngRates[:,0]),np.max(AngRates[:,0])]
    omgpitchlimits=[np.min(AngRates[:,1]),np.max(AngRates[:,1])]
    omgrolllimits=[np.min(AngRates[:,2]),np.max(AngRates[:,2])]
    
    dx=[0.1,0.1,0.1]
    
    
    
    def pose2Rt(x):
        t=x[0:3].copy()
        phi=x[3]
        xi=x[4]
        zi=x[5]
        
        Rzphi,dRzdphi=gicp.Rz(phi)
        Ryxi,dRydxi=gicp.Ry(xi)
        Rxzi,dRxdzi=gicp.Rx(zi)
        
        R = Rzphi.dot(Ryxi)
        R=R.dot(Rxzi)
        
        return R,t
    
    def Rt2pose(R,t):
        x=np.zeros(6)
        x[:3]=t
        r = Rsc.from_matrix(R)
        phidt,xidt,zidt =r.as_euler('zyx', degrees=False)
        x[3]=phidt
        x[4]=xidt
        x[5]=zidt
        return x
    
    # x is forward
    
    
    def dynmodelCT3D(xstatepf,dt):
        # xstatepf1=np.zeros_like(xstatepf)
        
        for i in range(xstatepf.shape[0]):
            xx=xstatepf[i]
            
            
            t=xx[0:3]
            phi=xx[3]
            xi=xx[4]
            zi=xx[5]
            
            v=xx[6]
            omgyaw = xx[7]
            omgpitch = xx[8]
            omgroll = xx[9]
            
            a=xx[10]
            
            Rzphi,dRzdphi=gicp.Rz(phi)
            Ryxi,dRydxi=gicp.Ry(xi)
            Rxzi,dRxdzi=gicp.Rx(zi)
            
            R = Rzphi.dot(Ryxi)
            R=R.dot(Rxzi)
            # R=nplinalg.inv(R)
            
            drn = R.dot([1,0,0])
            tdt=t+(v*drn)*dt+0.5*(a*drn)*dt**2
            vdt=v+a*dt
            
            phidt=phi+omgyaw*dt
            xidt=xi+omgpitch*dt
            zidt=zi+omgroll*dt
            
            
            # Rzphi,dRzdphi=gicp.Rz(omgyaw*dt)
            # Ryxi,dRydxi=gicp.Ry(omgpitch*dt)
            # Rxzi,dRxdzi=gicp.Rx(omgroll*dt)
            
            # Romg = Rzphi.dot(Ryxi)
            # Romg=Romg.dot(Rxzi)
            
            # Rdt=R.dot(Romg)
            
            # r = Rsc.from_matrix(Rdt)
            # phidt,xidt,zidt =r.as_euler('zyx', degrees=False)
            
            xstatepf[i,3]=phidt
            xstatepf[i,4]=xidt
            xstatepf[i,5]=zidt
            
            xstatepf[i,0:3]=tdt
            xstatepf[i,6]=vdt
    
        
        return xstatepf
    
    
    def dynmodelUM3D(xstatepf,dt):
        # xstatepf1=np.zeros_like(xstatepf)
        # xstatepf = [x,y,z,vx,vy,vz,ax,ay,az]
        for i in range(xstatepf.shape[0]):
            xx=xstatepf[i]
            
            
            t=xx[0:3]
            v=xx[3:6]
            a=xx[6:]
            
    
            tdt=t+v*dt+0.5*a*dt**2
            xstatepf[i,0:3]=tdt
            
            vdt = v+a*dt
            xstatepf[i,3:6]=vdt
            
            
            
            
    
        
        return xstatepf
    
              
    # car pose is phi,xi,zi,tpos,v,omgyaw,omgpitch,omgroll    where v is velocity of car, omg is angular velocity
    
    # initialize
    
    # CT
    def getCTinitialSamples_origin(Npf):
        xstatepf=np.zeros((Npf,11))
        
        
        xstatepf[:,0]=1*np.random.randn(Npf)
        xstatepf[:,1]=1*np.random.randn(Npf)
        xstatepf[:,2]=0.1*np.random.randn(Npf)
        
        #yaw
        xstatepf[:,3]=5*np.random.randn(Npf)
        #pitch
        xstatepf[:,4]=5*np.random.randn(Npf)
        #roll
        xstatepf[:,5]=0.001*np.random.randn(Npf)
        
        
        
        #vel
        xstatepf[:,6]=5*np.abs(np.random.randn(Npf))
        
        #omgyaw
        xstatepf[:,7]=0.001*np.random.randn(Npf)
        #omgpitch
        xstatepf[:,8]=0.001*np.random.randn(Npf)
        #omgroll
        xstatepf[:,9]=0.001*np.random.randn(Npf)
        
        #acc
        a=nplinalg.norm(Acc,axis=1)
        anormlimits=[np.min(a),np.max(a)]
        xstatepf[:,10]=0.01*np.abs(np.random.randn(Npf))
        
        wpf=np.ones(Npf)/Npf
        
        Q=np.diag([(0.5)**2,(0.5)**2,(0.1)**2, # x-y-z
                    (5*np.pi/180)**2,(2*np.pi/180)**2,(0.1*np.pi/180)**2, # angles
                    (0.1)**2,   # velocity
                    (0.05*np.pi/180)**2,(0.05*np.pi/180)**2,(0.02*np.pi/180)**2,# angle rates
                    (0.001)**2]) #acc
        
        return xstatepf,wpf,Q
    
    def getCTinitialSamples(Npf):
        xstatepf=np.zeros((Npf,11))
        
        
        xstatepf[:,0]=np.random.rand(Npf)*(Xlimits[1]-Xlimits[0])+Xlimits[0]
        xstatepf[:,1]=np.random.rand(Npf)*(Ylimits[1]-Ylimits[0])+Ylimits[0]
        xstatepf[:,2]=np.random.rand(Npf)*(Zlimits[1]-Zlimits[0])+Zlimits[0]
        
        #yaw
        xstatepf[:,3]=np.random.rand(Npf)*(yawlimits[1]-yawlimits[0])+yawlimits[0]
        #pitch
        xstatepf[:,4]=np.random.rand(Npf)*(pitchlimits[1]-pitchlimits[0])+pitchlimits[0]
        #roll
        xstatepf[:,5]=np.random.rand(Npf)*(rolllimits[1]-rolllimits[0])+rolllimits[0]
        
        
        
        #vel
        xstatepf[:,6]=np.random.rand(Npf)*(vlimits[1]-vlimits[0])+vlimits[0]
        
        #omgyaw
        xstatepf[:,7]=np.random.rand(Npf)*(omgyawlimits[1]-omgyawlimits[0])+omgyawlimits[0]
        #omgpitch
        xstatepf[:,8]=np.random.rand(Npf)*(omgpitchlimits[1]-omgpitchlimits[0])+omgpitchlimits[0]
        #omgroll
        xstatepf[:,9]=np.random.rand(Npf)*(omgrolllimits[1]-omgrolllimits[0])+omgrolllimits[0]
        
        #acc
        a=nplinalg.norm(Acc,axis=1)
        anormlimits=[np.min(a),np.max(a)]
        xstatepf[:,10]=np.random.rand(Npf)*(anormlimits[1]-anormlimits[0])+anormlimits[0]
        
        wpf=np.ones(Npf)/Npf
        
        Q=np.diag([(0.1)**2,(0.1)**2,(0.1)**2, # x-y-z
                    (2*np.pi/180)**2,(2*np.pi/180)**2,(0.1*np.pi/180)**2, # angles
                    (0.2)**2,   # velocity
                    (0.05*np.pi/180)**2,(0.05*np.pi/180)**2,(0.02*np.pi/180)**2,# angle rates
                    (0.01)**2]) #acc
        
        
        return xstatepf,wpf,Q
    
    # UM
    def getUMinitialSamples(Npf):
        xstatepf=np.zeros((Npf,9))
        
        
        xstatepf[:,0]=np.random.rand(Npf)*(Xlimits[1]-Xlimits[0])+Xlimits[0]
        xstatepf[:,1]=np.random.rand(Npf)*(Ylimits[1]-Ylimits[0])+Ylimits[0]
        xstatepf[:,2]=np.random.rand(Npf)*(Zlimits[1]-Zlimits[0])+Zlimits[0]
        
        #vx
        xstatepf[:,3]=np.random.rand(Npf)*(Vxlimits[1]-Vxlimits[0])+Vxlimits[0]
        #vy
        xstatepf[:,4]=np.random.rand(Npf)*(Vylimits[1]-Vylimits[0])+Vylimits[0]
        #vy
        xstatepf[:,5]=np.random.rand(Npf)*(Vzlimits[1]-Vzlimits[0])+Vzlimits[0]
        
        
        
        #ax
        xstatepf[:,6]=np.random.rand(Npf)*(Axlimits[1]-Axlimits[0])+Axlimits[0]
        #ay
        xstatepf[:,7]=np.random.rand(Npf)*(Aylimits[1]-Aylimits[0])+Aylimits[0]
        #az
        xstatepf[:,8]=np.random.rand(Npf)*(Azlimits[1]-Azlimits[0])+Azlimits[0]
        
        wpf=np.ones(Npf)/Npf
        
        Q=1*np.diag([(0.1)**2,(0.1)**2,(0.1)**2, # x-y-z
                    (1)**2,(1)**2,(0.25)**2, # vel
                    (0.5)**2,(0.5)**2,(0.025)**2]) # acc
        
        return xstatepf,wpf,Q
    
    
    
    
    
    def measModel(xx):
        t=xx[0:3]
        r=nplinalg.norm(t)
        th = np.arccos(t[2]/r)
        phi = np.arctan2(t[1],t[0])
        
        return np.array([r,th,phi])
    
    
    
    
    
    
    
    
    class plot3dPF:
        def __init__(self,Xtpath,Xmap2Dflat,saveplot=True):
            
            self.Xtpath=Xtpath.copy()
            self.time_taken=[]
            
    
            
            # self.figpf2Dzoom = plt.figure(figsize=(20, 20))    
            # self.axpf2Dzoom = self.figpf2Dzoom.add_subplot(111)
            
            self.Xmap2Dflat=Xmap2Dflat
            self.saveplot=saveplot
            
            
            self.xtpath2Dhandle=None
            self.pf2Dhandle=None
            self.ellipse2Dhandle=None
            self.arrow2Dhandle=[]
            self.binmatch2Dhandle=None
            
            # self.vis = o3d.visualization.Visualizer()
            # self.vis.create_window()
    
        def plotOpen3D(self,k,X,X1gv,sleep=0.5):
            pcdPF = o3d.geometry.PointCloud()
            pcdPF.points = o3d.utility.Vector3dVector(X[:,:3])
            pcdPF.paint_uniform_color([1,0,0]) #black
            
    
            # axpf.plot(G[:,0],G[:,1],G[:,2],'b')
            # axpf.plot(X1gv_pose[:,0],X1gv_pose[:,1],X1gv_pose[:,2],'r.')
            
            # pcdX1gv_pose = o3d.geometry.PointCloud()
            # pcdX1gv_pose.points = o3d.utility.Vector3dVector(X1gv_pose)
            # pcdX1gv_pose.paint_uniform_color([0,1,0]) #green
            
            bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(X1gv,axis=0)-15,max_bound=np.max(X1gv,axis=0)+15)
            pcdbox=pcd3DdownSensorCost.crop(bbox3d)
            pcdbox.paint_uniform_color([211/255,211/255,211/255])
            
            ArrXpf=[]
            for i in range(X.shape[0]):
                
                
                R,t = pose2Rt(X[i][:6])
                
                drn = R.dot([1,0,0])
                t2=t+5*drn
                ArrXpf.append(t2)
                
                
            # plot PF arrows as red and pose pf point as green
            ArrXpf=np.array(ArrXpf)
            pfline_set = o3d.geometry.LineSet()
            pfline_set.points = o3d.utility.Vector3dVector( np.vstack([X[:,:3],ArrXpf]))
            pflines = [[i,i+Npf] for i in range(Npf)]
            pfline_set.lines = o3d.utility.Vector2iVector(pflines)
            colors = [[1, 0, 0] for i in range(len(pflines))] # red
            pfline_set.colors = o3d.utility.Vector3dVector(colors)
            # np.asarray(pfline_set.colors)[0]=[0,1,0] #green
            
            
            # plot path as black
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(self.Xtpath[:k,:3])
            lines = [[i,i+1] for i in range(k-1)]
            line_set.lines = o3d.utility.Vector2iVector(lines)
            colors = [[0, 0, 0] for i in range(len(lines))]
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            # pcdbox.paint_uniform_color([0,0,0])
            o3d.visualization.draw_geometries([pcdbox,pcdPF,pfline_set]) #line_set
            # self.vis.update_geometry([pcdbox,pcdPF,pfline_set])
            # self.vis.update_renderer()
            
        def plot3D(self,k,X,m,P,sleep=0.5):
            st=time.time()
    
                
            figpf = plt.figure(figsize=(20, 20))    
            axpf = figpf.add_subplot(111,projection='3d')
            
            axpf.plot(self.Xtpath[k0:k,0],self.Xtpath[k0:k,1],self.Xtpath[k0:k,2],'b')
            axpf.plot(X[:,0],X[:,1],X[:,2],'r.')
    
            for i in range(X.shape[0]):
                # R,t = pose2Rt(robotpf.X[i][:6])
                HH=kittilocal.pose2Hmat(X[i][:6])
                R = HH[0:3,0:3]
                t = HH[0:3,3]
                
                drn = R.dot([1,0,0])
                t2=t+5*drn
                G=np.vstack([t,t2])
                axpf.plot(G[:,0],G[:,1],G[:,2],'r')
                
    
                
            axpf.set_title("time step = %d"%k)
            axpf.set_xlabel('x')
            axpf.set_ylabel('y')
            axpf.set_zlabel('z')
    
            
            et=time.time()
            
            self.time_taken.append(et-st)
            
            plt.show(block=False)
            plt.pause(sleep)
            if self.saveplot:
                figpf.show()
                simmanger.savefigure(figpf, ['3Dmap','snapshot'], 'snapshot_'+str(int(k)),data=[k,X,m,P])
    
            
        def plot2D(self,k,X,m,P,X1vroadrem,HHs,sleep=0.5):
            
                
            figpf2D = plt.figure(figsize=(20, 20))    
            axpf2D = figpf2D.add_subplot(111)
    
            axpf2D.cla()
            axpf2D.plot(self.Xmap2Dflat[:,0],self.Xmap2Dflat[:,1],'k.')
            # if self.xtpath2Dhandle is not None:
            #     K=self.xtpath2Dhandle.pop(0)
            #     K.remove()
                
            #     K=self.pf2Dhandle.pop(0)
            #     K.remove()
                
            #     K=self.ellipse2Dhandle.pop(0)
            #     K.remove()
                
            #     while(len(self.arrow2Dhandle)>0):
            #         K=self.arrow2Dhandle.pop(0)
            #         for ss in K:
            #             K.remove(ss)
                
            #     K=self.binmatch2Dhandle.pop(0)
            #     K.remove()
            
            
            axpf2D.plot(self.Xtpath[k0:k,0],self.Xtpath[k0:k,1],'b')
            axpf2D.plot(X[:,0],X[:,1],'r.')
            
            XX=utpltgmshp.getCovEllipsePoints2D(m[0:2],P[0:2,0:2],nsig=1,N=100)
            axpf2D.plot(XX[:,0],XX[:,1],'r--')
            
            for i in range(X.shape[0]):
                # R,t = pose2Rt(robotpf.X[i][:6])
                HH=kittilocal.pose2Hmat(X[i][:6])
                R = HH[0:3,0:3]
                t = HH[0:3,3]
                
                drn = R.dot([1,0,0])
                t2=t+5*drn
                G=np.vstack([t,t2])
     
                
                axpf2D.plot(G[:,0],G[:,1],'r')
                
            if HHs is not None:
                # idxpf,kcompleted,Hpose,gHkcorr=HHs
                idxpf,tk,kk,gHkest_initial,gHkcorr_attk,gHkcorr_atk,gHkcorr,solQt0,solQtf,solQtk = HHs
                
                
                pcd2ddown = o3d.geometry.PointCloud()
                X1vroadrem2Dflatcorr=gHkcorr_attk[0:3,0:3].dot(X1vroadrem.T).T+gHkcorr_attk[0:3,3]
                X1vroadrem2Dflatcorr[:,2]=0
                
                xpos_corr_atk = gHkcorr_atk[0:3,3]
                axpf2D.plot(xpos_corr_atk[0],xpos_corr_atk[1],'b^',markersize=12)
                
                pcd2ddown.points = o3d.utility.Vector3dVector(X1vroadrem2Dflatcorr[:,:3])
                pcd2ddown = pcd2ddown.voxel_down_sample(voxel_size=2)
                
                X1vroadrem2Dflatcorr = np.asarray(pcd2ddown.points)
                X1vroadrem2Dflatcorr=X1vroadrem2Dflatcorr[:,:2]
                axpf2D.plot(X1vroadrem2Dflatcorr[:,0],X1vroadrem2Dflatcorr[:,1],'g.')
                
            axpf2D.set_title("time step = %d"%k)
            axpf2D.set_xlabel('x')
            axpf2D.set_ylabel('y')
            axpf2D.axis('equal')
    
            plt.show(block=False)
            plt.pause(sleep)
            if self.saveplot:
                
                figpf2D.show()
                simmanger.savefigure(figpf2D, ['2Dmap','snapshot'], 'snapshot_'+str(int(k)),data=[k,X,m,P,HHs])
                
                xlim =[Xtpath[k,0]+np.min(X1vroadrem[:,0])-25,Xtpath[k,0]+np.max(X1vroadrem[:,0])+25]
                ylim =[Xtpath[k,1]+np.min(X1vroadrem[:,1])-25,Xtpath[k,1]+np.max(X1vroadrem[:,1])+25]
                
                xlim =[Xtpath[k,0]-75,Xtpath[k,0]+75]
                ylim =[Xtpath[k,1]-75,Xtpath[k,1]+75]
                
                
                axpf2D.set_xlim(xlim)
                axpf2D.set_ylim(ylim)
                # self.axpf2D.axis('equal')
                plt.show(block=False)
                plt.pause(0.1)
                figpf2D.show()
                simmanger.savefigure(figpf2D, ['2Dmap','snapshot_closeup'], 'snapshot_'+str(int(k)),data=[k,X,m,P,HHs])
        
    
    
    Rposnoise=np.diag([(1)**2,(1)**2,(1)**2])
        
    sampligfunc = getCTinitialSamples
    # sampligfunc= getCTinitialSamples_origin
    
    xstatepf, wpf, Q =sampligfunc(Npf)
    # xstatepf, wpf, Q =getUMinitialSamples(Npf)
    
    dim=xstatepf.shape[1]
    robotpf=pf.Particles(X=xstatepf,wts=wpf)
    
    
    
    
    plt.close("all")
    kittisimplotter = plot3dPF(Xtpath,Xmap2Dflat,saveplot=False)
    
    doneLocalize=0
    
    Hgt=dataset.calib.T_cam0_velo
    Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[k0].dot(Hgt))
    sm=KL.addMeas_fromQ(Hgt,dataset.times[k0])
    
    
    # dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem=getmeas(0)
    # KL.addMeas(X1v,X1v_roadrem,tk)
    
    
    m,P=robotpf.getEst()
    XPFmP_history=[(m,P)]
    XPFhistory=[(robotpf.X.copy(),robotpf.wts.copy())]
    simmanger.data['resampled?']={}
    simmanger.data['BinMatch_idxpf']=[]
    simmanger.data['BinMatchedH']={}
    isBMworking=False
    
    simmanger.data['doneLocalize']={}
    simmanger.data['lost']={}
    simmanger.data['reinitialized?']={}
    
    HHs=None
    kittisimplotter.plot2D(k0,robotpf.X,m,P,sm.X1v_roadrem,HHs,sleep=0.1)
    
    simmanger.data['vehicle_status']={}
    vehicle_status = simmanger.data['vehicle_status'][k0] = "Localization"
        
    
    fps=time.time()
    simmanger.data['fps']=[]
    
    simmanger.data['simtvec']=[dataset.times[k0]]
    
    [Xbm,Ybm]=np.meshgrid(np.linspace(Xlimits[0], Xlimits[1],5),np.linspace(Ylimits[0], Ylimits[1],5))
    XXbm=np.vstack([Xbm.reshape(-1),Ybm.reshape(-1)]).T
    idxx=list(range(XXbm.shape[0]))
    np.random.shuffle( idxx)
    XXbm=XXbm[idxx,:]
    nbm = XXbm.shape[0]
    ccbm=0
    Pbm=np.ones(nbm)
    
    for k in range(k0+1,len(dataset)):
        simmanger.data['fps'].append( time.time()-fps  )
        fps=time.time()
        simmanger.data['simtvec'].append(dataset.times[k])
        
        print(k)
        st=time.time()
    
        
        st=time.time()
        Hgt=dataset.calib.T_cam0_velo
        Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[k].dot(Hgt))
        sm=KL.addMeas_fromQ(Hgt,dataset.times[k])
        dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem = sm.dt,sm.tk,sm.X1v,sm.X1gv,sm.X1v_roadrem,sm.X1gv_roadrem
        et=time.time()
        print("get data time = ",et-st)
    
        st=time.time()
        KL.setRegisteredSeqH_async()
        et=time.time()
        print("setRegisteredSeqH_async time = ",et-st)
    
        
        
    
        # propagate
        st=time.time()
        robotpf.X=dynmodelCT3D(robotpf.X,dt)
        
        
        
        et=time.time()
        print("dyn model time = ",et-st)
        robotpf.X=robotpf.X+1.5*np.random.multivariate_normal(np.zeros(dim), Q, robotpf.X.shape[0])
        
    
    
        ### BIN MATCH
        
            
        # idxpf=np.random.choice(list(range(Npf)),1,p=robotpf.wts)
        # idxpf=idxpf[0]
        
        # idxpf=np.argmin(robotpf.wts)
        
    
        simmanger.data['doBinMatch']=doBinMatch=1
        
        KL.setRelStates_async()
        
        if doBinMatch==1 and isBMworking==False:
            if vehicle_status=="Localization" or vehicle_status=="Tracking":
                # mm=np.mean(robotpf.wts)
                # idxpf = (np.abs(robotpf.wts - mm)).argmin()
                idxpf=np.argmin(robotpf.wts)
                
                # if np.all(Pbm>0):
                if np.all(Pbm==0):
                    Pbm[Pbm==0]=1
                ccbm=np.random.choice(range(nbm),size=1, replace=False, p=Pbm/np.sum(Pbm))[0]
                robotpf.X[idxpf,0]=XXbm[ccbm,0]
                robotpf.X[idxpf,1]=XXbm[ccbm,1]
                # else:
                #     robotpf.X[idxpf,0]=XXbm[ccbm,0]
                #     robotpf.X[idxpf,1]=XXbm[ccbm,1]
                #     ccbm+=1
                    
                # if ccbm==nbm:
                #     Pbm[Pbm==0]=1
                #     idxx=list(range(XXbm.shape[0]))
                #     np.random.shuffle( idxx)
                #     XXbm=XXbm[idxx,:]
                #     Pbm=Pbm[idxx]
                #     ccbm=0
                    
                DD=copy.deepcopy(D)
                # DD['BinMatch']["dxMatch"]=list(np.array([1,1],dtype=np.float64))
                DD['BinMatch']["dxBase"]=list(np.array([30,30],dtype=np.float64))
                DD['BinMatch']["Lmax"]=list(np.array([200,200],dtype=np.float64))
                KL.setOptions_noreset(json.dumps(DD))
    
            # if vehicle_status=="Tracking":
            #     # kmeans = KMeans(n_clusters=5, random_state=10).fit(robotpf.X[:,:2])
            #     # kmeans.cluster_centers_
                
            #     idxpf=np.argmax(robotpf.wts)
            #     DD=copy.deepcopy(D)
            #     # DD['BinMatch']["dxMatch"]=list(np.array([1,1],dtype=np.float64))
            #     DD['BinMatch']["dxBase"]=list(np.array([25,25],dtype=np.float64))
            #     DD['BinMatch']["Lmax"]=list(np.array([100,100],dtype=np.float64))
            #     KL.setOptions_noreset(json.dumps(DD))
                
            print("---doneLocalize ------ = ",doneLocalize)
            simmanger.data['BinMatch_idxpf'].append((k,idxpf))
            # Ridx,tidx = pose2Rt(robotpf.X[idxpf][:6])
            Hpose = kittilocal.pose2Hmat(robotpf.X[idxpf][:6])
            xpose_k = robotpf.X[idxpf].copy()
            xpose_k[-1]=0
            t0=dataset.times[k]  #max([0,k-5])max([k0,k-10])
            tf=dataset.times[k]
            tk=dataset.times[k]
            
            # st=time.time()
            # solret=KL.BMatchseq(t0,tf,tk,Hpose,True)
            # et=time.time()
            # print("Bin match and gicp time = ",et-st)
            
            KL.BMatchseq_async(t0,tf,tk,Hpose,True)
            isBMworking=True
            time.sleep(0.2)
            # et=time.time()
            # print("Bin match asyc set time = ",et-st)
        
    
        # start computation of relative states
        KL.setRelStates_async()
        
        solQ=KL.getBMatchseq_async()
    
        if solQ.isDone==True:
            isBMworking=False
            solret=solQ.bmHsol
            print("-----------SOL IS DONE - -----------------------------")
            print("-----------SOL IS DONE - -----------------------------")
            # idxpfmin = np.argmin(robotpf.wts)
            # robotpf.X[idxpfmin]=robotpf.X[idxpf].copy()
            # xpose_tk=robotpf.X[idxpf].copy()
            time.sleep(0.25)
            gHkseq=KL.getSeq_gHk()
            Velmeas=KL.getvelocities()
            AngVelmeas=KL.getangularvelocities()
            fflg=0
            for si in range(len(solret.gHkcorr)):
                print(solret.sols[si].costfrac)
                if solret.sols[si].cost>solret.sols[si].cost0 and solret.sols[si].costfrac>0.5:
                    # Pbm[ccbm]+=2
                    
                    
                    xpose_tk=xpose_k.copy()
                    
                    tkidx_datasettimes=np.argwhere(np.isclose(dataset.times,solQ.tk,rtol=1e-05, atol=1e-05))[0][0]
                    
                    tkidx = KL.time2index(solQ.tk)
                    gHtk=gHkseq[tkidx].copy()
                    gHk=gHkseq[-1].copy()
                    
                    tkHg=nplinalg.inv(gHtk)
                    kHtk=gHk.dot(tkHg)
                    gHkcorr_atk = solret.gHkcorr[si].dot(kHtk)
                    gHkcorr_attk = solret.gHkcorr[si]
                    
                    DD=copy.deepcopy(D)
                    # DD['BinMatch']["dxMatch"]=list(np.array([1,1],dtype=np.float64))
                    DD['BinMatch']["dxBase"]=list(np.array([10,10],dtype=np.float64))
                    DD['BinMatch']["Lmax"]=list(np.array([20,20],dtype=np.float64))
                    KL.setOptions_noreset(json.dumps(DD))
    
    
                    tkk=dataset.times[k]  
                    st=time.time()
                    solret2=KL.BMatchseq(tkk,tkk,tkk,gHkcorr_atk,True)
                    et=time.time()
                    print("Bin match and gicp time = ",et-st)
                    if solret2.sols[0].cost>solret2.sols[0].cost0 and solret2.sols[0].costfrac>0.6:
                        pass
                    else:
                        continue
                    fflg=1
                    
                    gHkcorr_atk=solret2.gHkcorr[0]
                    xpose_tk[0:6] = kittilocal.Hmat2pose(gHkcorr_atk)
                    
                    # xpose_tk[0:6]=KL.gicp_correction_pose(dataset.times[k],xpose_tk[0:6])
                    
                    xpose_tk[6]=nplinalg.norm(Velmeas[-1])
                    xpose_tk[7:10]=AngVelmeas[-1]
                    
                    Psamp = np.diag([50*nplinalg.norm(Velmeas[-1])*dt,7,7])
                    Xsamp = np.random.multivariate_normal(np.zeros(3), Psamp, 100)
                    
                    Xnewsamp = np.tile(xpose_tk, [100,1])
                    Xnewsamp[:,0:3]=gHkcorr_atk[0:3,0:3].dot(Xsamp.T).T+gHkcorr_atk[0:3,3]
                    
                    robotpf.X =np.vstack([robotpf.X,xpose_tk.copy(),Xnewsamp])
                    robotpf.wts = np.hstack([robotpf.wts,1*np.max(robotpf.wts)*np.ones(Xnewsamp.shape[0]+1)])
                    
                    
                    # likelihood=KL.getLikelihoods_lookup(np.array([xpose_tk]),dataset.times[k])
    
            
                    if si==0:
                        simmanger.data['BinMatchedH'][k]=(idxpf,tkidx_datasettimes,k,solQ.gHkest_initial,gHkcorr_attk,gHkcorr_atk,solret.gHkcorr,solQ.t0,solQ.tf,solQ.tk)    
    
            if fflg==1:
                Pbm[ccbm]+=5
                Pbm[ccbm]=np.min([Pbm[ccbm],20])
            else:
                Pbm[ccbm]=1
                Pbm[ccbm] = np.max([Pbm[ccbm],1])
                
            # break
            if fflg==1:
                robotpf.wts=robotpf.wts/np.sum(robotpf.wts)            
    
        # measurement update
        
        if vehicle_status == "Localization":
            DD=copy.deepcopy(D)
            DD["Localize"]["sig0"]=0.5
            KL.setOptions_noreset(json.dumps(DD))
        
        if vehicle_status == "Tracking":
            KL.setOptions_noreset(json.dumps(D))
        
        KL.setRelStates_async()
        
        
        st=time.time()
        # likelihoods_octree=KL.getLikelihoods_octree(robotpf.X,dataset.times[k])
        # likelihoods=KL.getLikelihoods_octree(robotpf.X,dataset.times[k])
        likelihoods=KL.getLikelihoods_lookup(robotpf.X,dataset.times[k])
        
        likelihoods_exp=np.exp(-likelihoods)
        likelihoods_exp[likelihoods_exp<1e-190]=1e-190
        likelihoods_exp[np.isnan(likelihoods_exp)]=1e-190
        
        lostflg = np.all(likelihoods_exp<=1e-170)
        
        robotpf.wts=likelihoods_exp*robotpf.wts
        et=time.time()
        print("pcl meas model time = ",et-st)
        
        robotpf.renormlizeWts()
        
        # lost target
        if lostflg:
            simmanger.data['doneLocalize'][k]=doneLocalize=0
            xstatepf, wpf, Q =sampligfunc(Npf)
            robotpf.X=xstatepf
            robotpf.wts=wpf
            
            simmanger.data['lost'][k]=1
            simmanger.data['reinitialized?'][k]=1
    
            vehicle_status = simmanger.data['vehicle_status'][k] = "Localization"
        
        if np.any(np.isnan(robotpf.wts)):
            robotpf.wts[np.isnan(robotpf.wts)]=0
            robotpf.renormlizeWts()
        
        simmanger.data['vehicle_status'][k]=vehicle_status
        
        
        ### boostratp  resample 
        
        Neff=robotpf.getNeff()
        
        if vehicle_status=="Localization":
            simmanger.data['Neff/Npf']=0.1
            if Neff/Npf<simmanger.data['Neff/Npf']:
                print("resampled at k = ",k)
                # robotpf.bootstrapResample(Npf=Npf,fraclowH=0.1,replace=False)
                # robotpf.bootstrapResample_vanilla(Npf=Npf)
                robotpf.bootstrapResample_clustered(Nclusters=5,fraclowH=0.01,Npf=Npf)
                simmanger.data['resampled?'][k]=1
            # Npf = 100
                
        if vehicle_status=="Tracking":
            simmanger.data['Neff/Npf']=0.75
            if Neff/Npf<simmanger.data['Neff/Npf']:
                print("resampled at k = ",k)
                robotpf.bootstrapResample_vanilla(Npf=Npf)
                simmanger.data['resampled?'][k]=1
            # Npf = simmanger.data['Npf']=D['PF']['Npf']
    
        
        m,P=robotpf.getEst()
        u,v=nplinalg.eig(P)
        
        pos_std=np.sqrt(u[:3])
        if max(pos_std)<10:
            simmanger.data['doneLocalize'][k]=doneLocalize=1
            vehicle_status = simmanger.data['vehicle_status'][k] = "Tracking"
        
            
        # lost target
        if min(pos_std)>200:
            simmanger.data['doneLocalize'][k]=doneLocalize=0
            vehicle_status = simmanger.data['vehicle_status'][k] = "Localization"
            
        
        
    
    
        ### saving and plotting  resample 
        XPFhistory.append((robotpf.X.copy(),robotpf.wts.copy()))
        
        # robotpf.regularizedresample(kernel='gaussian',scale=1/25)
        
        XPFmP_history.append((m,P))
        
        if (k%25==0 or k==len(dataset)-1 or k in simmanger.data['BinMatchedH'].keys()):
        # if (k in simmanger.data['BinMatchedH'].keys()):    
        # if (k%25==0 or k==len(dataset)-1):
            # kittisimplotter.plot3D(k,robotpf.X,m,P)
            plt.close("all")
            if k in simmanger.data['BinMatchedH'].keys():
                HHs=simmanger.data['BinMatchedH'][k]
            else:
                HHs=None
            st=time.time()
            kittisimplotter.plot2D(k,robotpf.X,m,P,X1v_roadrem,HHs,sleep=0.1)
            et=time.time()
            print("plot time = ",et-st)
            # kittisimplotter.plotOpen3D(k,robotpf.X,X1gv)
            # plt.show(block=True)
            # plt.pause(1.5)
            
    
            
    
        # plt.show()
        # plt.pause(0.1)
        
        # if k>=400:
        #     break
    
    KL.setRegisteredSeqH_async()
    time.sleep(2)
    KL.setRelStates_async()
    
    
    mest = np.array([x[0] for x in XPFmP_history])
    Pest = np.array([x[1] for x in XPFmP_history])
    Peststd=[]
    for i in  range(Pest.shape[0]):
        u,v = nplinalg.eig(Pest[i])
        Peststd.append(u)    
    Peststd=np.array(Peststd)
    simtvec=simmanger.data['simtvec']=np.array(simmanger.data['simtvec'])
    
    vehicle_status_vec = [(k,v) for k,v in simmanger.data['vehicle_status'].items()]
    sorted(vehicle_status_vec,key=lambda x: x[0])
    vehicle_status_vec=[x[1] for x in vehicle_status_vec]
    vehicle_status_vec=np.array([s=='Tracking' for s in vehicle_status_vec],dtype=bool)
    simmanger.data['vehicle_status_vec']=vehicle_status_vec
    
    
    timers=KL.gettimers()
    
    Velmeas=KL.getvelocities()
    AngVelmeas=KL.getangularvelocities()
    PosRelmeas=KL.getpositions()
    
    Velmeas=np.vstack(Velmeas)
    AngVelmeas=np.vstack(AngVelmeas)
    PosRelmeas=np.vstack(PosRelmeas)
    
    nt = PosRelmeas.shape[0]
    splx=UnivariateSpline(dataset.times[k0:],PosRelmeas[:,0])
    splvx=splx.derivative()
    splax=splvx.derivative()
    
    sply=UnivariateSpline(dataset.times[k0:],PosRelmeas[:,1])
    splvy=sply.derivative()
    splay=splvy.derivative()
    
    splz=UnivariateSpline(dataset.times[k0:],PosRelmeas[:,2])
    splvz=splz.derivative()
    splaz=splvz.derivative()
    
    Velocities_meas=np.zeros((nt,3))
    Velocities_meas[:,0]=splvx(dataset.times[k0:])
    Velocities_meas[:,1]=splvy(dataset.times[k0:])
    Velocities_meas[:,2]=splvz(dataset.times[k0:])
    
    plt.figure()
    plt.plot(dataset.times,Velocities[:,0])
    plt.plot(dataset.times[k0:],Velocities_meas[:,0],'r')
    plt.plot(dataset.times[k0:],Velmeas[:,0],'g')
    
    
    plt.figure()
    plt.plot(dataset.times,AngRates[:,0])
    plt.plot(dataset.times[k0:],AngVelmeas[:,0],'r')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,8],'g')
    
    fig=plt.figure()
    plt.plot(dataset.times,Xtpath[:,0],'r',label='ground truth')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,0],'b',label='estimate')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,0]+3*Peststd[vehicle_status_vec==1,0],'g--',label='estimate+3\sigma')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,0]-3*Peststd[vehicle_status_vec==1,0],'g--',label='estimate-3\sigma')
    plt.xlabel('time (s)')
    plt.ylabel('x position (m)')
    plt.legend()
    simmanger.savefigure(fig, ['Metrics'], 'Xpositionestimate',data=[])
    
    
    fig=plt.figure()
    plt.plot(dataset.times,Xtpath[:,1],'r',label='ground truth')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,1],'b',label='estimate')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,1]+3*Peststd[vehicle_status_vec==1,1],'g--',label='estimate+3\sigma')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,1]-3*Peststd[vehicle_status_vec==1,1],'g--',label='estimate-3\sigma')
    plt.xlabel('time (s)')
    plt.ylabel('y position (m)')
    plt.legend()
    simmanger.savefigure(fig, ['Metrics'], 'Ypositionestimate',data=[])
    
    fig=plt.figure()
    plt.plot(dataset.times,Xtpath[:,2],'r',label='ground truth')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,2],'b',label='estimate')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,2]+3*Peststd[vehicle_status_vec==1,2],'g--',label='estimate+3\sigma')
    plt.plot(simtvec[vehicle_status_vec==1],mest[vehicle_status_vec==1,2]-3*Peststd[vehicle_status_vec==1,2],'g--',label='estimate-3\sigma')
    plt.xlabel('time (s)')
    plt.ylabel('z position (m)')
    plt.legend()
    simmanger.savefigure(fig, ['Metrics'], 'Zpositionestimate',data=[])
    
    vehiclestates=[Velmeas,AngVelmeas,PosRelmeas]
    
    simmanger.finalize()
    
    simmanger.save(metalog, mainfile=runfilename, vehiclestates=vehiclestates,
                   options=D, Npf=Npf,XPFhistory=XPFhistory,
                   XPFmP_history=XPFmP_history,timers=timers)

sys.exit(0)
    #%%
    
    # gHk=KL.getSeq_gHk()
    # i1Hi_seq=KL.geti1Hi_seq()
    
    # cgHk=[np.identity(4)]
    # tpos=[[0,0,0]]
    # for i in range(len(i1Hi_seq)):
    #     iHi1=nplinalg.inv(i1Hi_seq[i])
    #     cgHk.append(cgHk[i].dot(iHi1))
    #     tpos.append(cgHk[-1][0:3,3])
    # tpos=np.vstack(tpos)
    # #%%
    # Velmeas=KL.getvelocities()
    # AngVelmeas=KL.getangularvelocities()
    # PosRelmeas=KL.getpositions()
    
    # Velmeas=np.vstack(Velmeas)
    # AngVelmeas=np.vstack(AngVelmeas)
    # PosRelmeas=np.vstack(PosRelmeas)
    
    # figposrel = plt.figure()    
    # axposrel = figposrel.add_subplot(111,projection='3d')
    # axposrel.plot(PosRelmeas[:,0],PosRelmeas[:,1],PosRelmeas[:,2],'b.')
    # axposrel.set_title("Rel Position Measurement")
    # axposrel.set_xlabel('x')
    # axposrel.set_ylabel('y')
    # axposrel.set_zlabel('z')
    # plt.pause(0.1)
    # # plt.show()
    
    # #%%
    # gHkss=KL.getsetSeq_gHk(dataset.times[1], np.identity(4))
    # Xmm=KL.getalignSeqMeas_eigen(0,0,0, np.identity(4),[0.2,0.2,0.2],3)
    # # pcd = o3d.geometry.PointCloud()
    # # pcd.points = o3d.utility.Vector3dVector(Xmm)
    
    # # o3d.visualization.draw_geometries([pcd])
    
    # Ridx,tidx = pose2Rt(robotpf.X[idxpf][:6])
    # Hpose = kittilocal.pose2Hmat(robotpf.X[idxpf][:6])
    # xpose = kittilocal.Hmat2pose(Hpose)
    
    # # #%% Bin match
    # idxpf=1
    # xx=robotpf.X[idxpf][:6].copy()
    # xx[:2]=xx[:2]+15
    
    # # Hpose=kittilocal.pose2Hmat(xx)
    # # Ridx = Hpose[0:3,0:3]
    # # tidx = Hpose[0:3,3]
    
    # t0=5
    # tf=5
    # tk=5
    
    # Hgt=dataset.calib.T_cam0_velo
    # Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[tk].dot(Hgt))
    
    # # DD={}
    # # DD["BinMatch"]={}
    # # DD['BinMatch']["dxMatch"]=list(np.array([1,1],dtype=np.float64))
    # # DD['BinMatch']["dxBase"]=list(np.array([30,30],dtype=np.float64))
    # # DD['BinMatch']["Lmax"]=list(np.array([200,200],dtype=np.float64))
    # # DD['BinMatch']["thmax"]=170*np.pi/180
    # # DD['BinMatch']["thfineres"]=2.5*np.pi/180
    
    # st=time.time()
    # solret=KL.BMatchseq(t0,tf,tk,Hpose,True)
    # et=time.time()
    # print("Bin match and gicp time = ",et-st)
    
    # Xmap2Dcpp = KL.getmap2D_noroad_res_eigen([0.25,0.25,1],2)
    # # Xmeasnorad=KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, Hpose,[D["BinMatch"]["dxMatch"][0],D["BinMatch"]["dxMatch"][1],1],2)
    
    # # bm=kittilocal.BinMatch(json.dumps(DD))
    # # bm.computeHlevels(Xmap2Dcpp)
    
    # # # Ridx,tidx = KL.pose2Rt(robotpf.X[idxpf][:6])
    # # phiidx = xx[3]
    # # tidx = xx[0:3]
    # # Rzphiidx,_=gicp.Rz(phiidx)
    
    # # H12est=np.identity(3)
    # # H12est[0:2,0:2]=Rzphiidx[0:2,0:2]
    # # H12est[0:2,2]=tidx[:2]
    # # sol=bm.getmatch(Xmeasnorad,H12est)
    # # Rt=np.identity(3)
    # # tr=np.zeros(3)
    # # Rt[0:2,0:2]=sol[0].H[0:2,0:2]
    # # tr[0:2]=sol[0].H[0:2,2]
    # # xc= Rt2pose(Rt,tr)
    
    # # xc[2]=xx[2]
    # # xc[4:]=xx[4:]
    # # Rc,tc=pose2Rt(xc)
    
    # Xmeas2Dcpp_pose = KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, Hpose,[0.25,0.25,1],2)
    # Xmeas2Dcpp_posecorrect = KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, solret.gHkcorr,[0.25,0.25,1],2)
    # Xmeas2Dcpp_posegt = KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, Hgt,[0.25,0.25,1],2)
    # # Xmeas2Dcpp_poseBMcorrect = KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, sol[0].H,[0.25,0.25,1],2)
    
    # lb=np.array([-88.1277 ,-54.7676, -15.1528])
    # ub=np.array([90.412, 86.3891, 16.6382 ])
    # Xlmp=KL.getmaplocal_eigen(lb,ub)
    # res=D["mapfit"]["downsample"]["resolution"]
    # Xmeastk=KL.getalignSeqMeas_eigen(t0,tf,tk, solret.gHkcorr,res,3)
    
    # figbf = plt.figure("bin-fit")
    # ax=figbf.add_subplot(111)
    # ax.cla()
    # ax.plot(Xmap2Dcpp[:,0],Xmap2Dcpp[:,1],'k.')
    # ax.plot(Xmeas2Dcpp_pose[:,0],Xmeas2Dcpp_pose[:,1],'b.',label='pf-pose')
    # ax.plot(Xmeas2Dcpp_posecorrect[:,0],Xmeas2Dcpp_posecorrect[:,1],'r.',label='pf-pose-corrected')
    # # ax.plot(Xmeas2Dcpp_poseBMcorrect[:,0],Xmeas2Dcpp_poseBMcorrect[:,1],'y.',label='pf-pose-BM-corrected')
    # ax.plot(Xmeas2Dcpp_posegt[:,0],Xmeas2Dcpp_posegt[:,1],'g.',label='ground truth')
    # ax.legend()
    # ax.axis("equal")
    
    
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(Xlmp)
    # pcd.paint_uniform_color([1,0,0]) #green
    
    # pcdmeastk = o3d.geometry.PointCloud()
    # pcdmeastk.points = o3d.utility.Vector3dVector(Xmeastk)
    # pcdmeastk.paint_uniform_color([0,1,0]) #green
    
    # o3d.visualization.draw_geometries([pcd,pcdmeastk])
    
    # #%%
    
    # st=time.time()
    # KL.BMatchseq_async(t0,tf,tk,Hpose,True)
    # et=time.time()
    # print("Bin match asyc set time = ",et-st)
    
    # solQ=KL.getBMatchseq_async()


#%% Summary plots


# with open('kitti_localize_config.yml', 'r') as outfile:
#     D=yaml.safe_load( outfile)

# dataset = odometry.odometry(D['Data']['folder'], D['Data']['sequence'], frames=None) # frames=range(0, 20, 5)
 
# nd=len(dataset.poses)

# Xtpath=np.zeros((len(dataset),10))

# for i in range(len(dataset)):
#     H=dataset.calib.T_cam0_velo
#     H=np.dot(nplinalg.inv(H),dataset.poses[i].dot(H))
    
#     Xtpath[i,0:3] = H.dot(np.array([0,0,0,1]))[0:3]
#     # Xtpath[i,0:3] = dataset.poses[i].dot(np.array([0,0,0,1]))[0:3]
    
#     r = Rsc.from_matrix(H[0:3,0:3])
    
#     q =r.as_euler('zyx',degrees=False)
#     Xtpath[i,3:6] = q
    
#     q =r.as_quat()
#     Xtpath[i,6:] = q

# tvec,qvec_sp,w,qdot,alpha=quat2omega_scipyspline(dataset.times,Xtpath[:,6:],k=3,s=0.0001)


    
# Velocities=np.zeros((len(dataset),3))
# Acc=np.zeros((len(dataset),3))

# splx=UnivariateSpline(dataset.times,Xtpath[:,0])
# splvx=splx.derivative()
# splax=splvx.derivative()

# sply=UnivariateSpline(dataset.times,Xtpath[:,1])
# splvy=sply.derivative()
# splay=splvy.derivative()

# splz=UnivariateSpline(dataset.times,Xtpath[:,2])
# splvz=splz.derivative()
# splaz=splvz.derivative()


# Velocities[:,0]=splvx(dataset.times)
# Velocities[:,1]=splvy(dataset.times)
# Velocities[:,2]=splvz(dataset.times)

# Acc[:,0]=splax(dataset.times)
# Acc[:,1]=splay(dataset.times)
# Acc[:,2]=splaz(dataset.times)

# rotations = Rsc.from_euler('zyx', Xtpath[:,3:6], degrees=False)
# spline =RscSpl(dataset.times, rotations)
# AngRates=spline(dataset.times, 1)
# AngAcc=spline(dataset.times, 2)
# AngRates=AngRates[:,::-1]
# AngAcc=AngAcc[:,::-1]

# # metrics
# # - 
# # - time steps taken to begin tracking (with in 3sigma of truth for 10 steps)
# # - error in "tracking", covariance of tracking
# # - 


# # simanger.data.keys()
# # Out[3]: dict_keys(['sequence', 'basedir', 'Npf', 'k0', 'resampled?', 'BinMatch_idxpf', 'BinMatchedH', 'doneLocalize', 'lost', 'reinitialized?', 'vehicle_status', 'fps', 'simtvec', 'doBinMatch', 'Neff/Npf', 'vehicle_status_vec'])

# # data.keys()
# # Out[4]: dict_keys(['mainfile', 'vehiclestates', 'options', 'Npf', 'XPFhistory', 'XPFmP_history', 'timers'])


# simfolder = '/media/na0043/misc/DATA/localization-paper'
# simfiles=os.listdir(simfolder)
# MTS={'fracstepstrack':[],'fracsteps2firsttrack':[],'fracstepstrack5mTruth':[],
#      'Nstepstrack':[],'Nsteps2firsttrack':[],'Nstepstrack5mTruth':[],
#      'rmsepos':[],'rmsedir':[]}
# FPS=[]
# plt.close("all")

# for ff in simfiles:
#     print(ff)
#     ff=os.path.join(simfolder,ff)
#     simanger,data=uqsimmanager.SimManager.load(ff)
    
#     simtvec = simanger.data['simtvec']
#     simtKvec = np.array(list(range(simanger.data['k0'],len(dataset.times))),dtype=int)
#     if 'vehicle_status_vec' not in simanger.data.keys():
#         vehicle_status_vec = [(k,v) for k,v in simanger.data['vehicle_status'].items()]
#         sorted(vehicle_status_vec,key=lambda x: x[0])
#         vehicle_status_vec=[x[1] for x in vehicle_status_vec]
#         vehicle_status_vec=np.array([s=='Tracking' for s in vehicle_status_vec],dtype=bool)
#     else:
#         vehicle_status_vec = simanger.data['vehicle_status_vec']
        
#     # designate tracking when eig(P)<5m and/or mu+-3sig
    
#     mest = np.array([x[0] for x in data['XPFmP_history']])
#     Pest = np.array([x[1] for x in data['XPFmP_history']])
    
    
    
#     ## time-steps taken to localize
#     fracstepstrack=np.round(100*np.sum(vehicle_status_vec==False)/len(vehicle_status_vec))
#     fracsteps2firsttrack = np.round(100*(np.argwhere(vehicle_status_vec==True)[0][0])/len(simtKvec))
    
#     MTS['Nstepstrack'].append(np.sum(vehicle_status_vec==False))
#     MTS['Nsteps2firsttrack'].append(np.argwhere(vehicle_status_vec==True)[0][0])
    
#     MTS['fracstepstrack'].append(fracstepstrack)
#     MTS['fracsteps2firsttrack'].append(fracsteps2firsttrack)
#     for i in range(len(simtKvec)):
#         if np.all(nplinalg.norm(Xtpath[simtKvec[i:i+10],0:3]-mest[i:i+10,0:3],axis=1)<5):
#             break
#     MTS['fracstepstrack5mTruth'].append(np.round(100*i/len(vehicle_status_vec)))
#     MTS['Nstepstrack5mTruth'].append(i)
    
#     ## error estimate tracking
#     TvecKTrack = simtKvec[vehicle_status_vec==True]
#     mestTrack = mest[vehicle_status_vec==True,:]
#     PestTrack = Pest[vehicle_status_vec==True,:]
#     simtvecTrack = simtvec[vehicle_status_vec==True]
#     EIG=[]
#     for i in range(len(PestTrack)):
#         u,v = nplinalg.eig(PestTrack[i])
#         EIG.append(u)
#     stdTrack=np.sqrt(np.array(EIG))
#     XtpathTrack = Xtpath[simanger.data['k0']:,:]
#     XtpathTrack = XtpathTrack[vehicle_status_vec==True,:]
#     rmsepos = np.sqrt(np.mean(np.sum(np.power(XtpathTrack[:,0:3]-mestTrack[:,0:3],2),axis=1)))
#     # rmseeuler = np.sqrt(np.mean((np.power(XtpathTrack[:,3]-mestTrack[:,3],2))))
#     # rmseeuler = np.sqrt(np.mean((np.power(XtpathTrack[:,4]-mestTrack[:,4],2))))
    
#     rotations_true = Rsc.from_euler('zyx', XtpathTrack[:,3:6], degrees=False)
#     rotations_est = Rsc.from_euler('zyx', mestTrack[:,3:6], degrees=False)
    
#     rotmattrue=rotations_true.as_matrix()
#     rotmatest=rotations_est.as_matrix()
    
#     rmsedir=np.sqrt(np.mean(np.sum(np.power(rotmattrue.dot([1,0,0])-rotmatest.dot([1,0,0]),2),axis=1)))
    
#     MTS['rmsepos'].append(rmsepos)
#     MTS['rmsedir'].append(rmsedir)
    
#     FPS+=simanger.data['fps']
    
    
    
    
# MTS=pd.DataFrame(MTS)

# ##
# methodstrings=['fracstepstrack','fracsteps2firsttrack','fracstepstrack5mTruth']
# xpos1=np.arange(len(methodstrings),dtype=np.int)

# fig1, ax1= plt.subplots(figsize=(20, 9))
# ax1.set_ylabel('percent')
# ax1.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
# ax1.set_xticklabels(['A','B','C'],fontsize=20)
# ax1.set_yticklabels(range(0,100,10),fontsize=20)
# ax1.set_ylabel("% of time steps",fontsize=20)
# # ax3.set_xlabel('Methods')
# ax1.grid(True,which="both",axis="y")

# ##
# methodstrings=['Nstepstrack','Nsteps2firsttrack','Nstepstrack5mTruth']
# xpos1=np.arange(len(methodstrings),dtype=np.int)

# fig0, ax0= plt.subplots(figsize=(20, 9))
# ax0.set_ylabel('percent')
# ax0.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
# ax0.set_xticklabels(['A','B','C'],fontsize=20)
# ax0.set_yticklabels(range(0,100,10),fontsize=20)
# ax0.set_ylabel("# scans",fontsize=20)
# # ax3.set_xlabel('Methods')
# ax0.grid(True,which="both",axis="y")


# ##
# methodstrings=['rmsepos']
# xpos1=np.arange(len(methodstrings),dtype=np.int)

# fig3, ax3= plt.subplots(figsize=(20, 9))
# ax3.set_ylabel('percent')
# # ax3.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
# MTS.hist(methodstrings,ax=ax3,xlabelsize=24,ylabelsize=24,bins=20)
# # ax3.set_xticklabels(methodstrings)
# ax3.set_xlabel("RMSE",fontsize=20)
# ax3.set_ylabel("percent",fontsize=20)
# # ax3.set_xlabel('Methods')
# ax3.grid(True,which="both",axis="y")

# ##
# methodstrings=['rmsedir']
# xpos1=np.arange(len(methodstrings),dtype=np.int)

# fig4, ax4= plt.subplots(figsize=(20, 9))
# ax4.set_ylabel('percent')
# # ax3.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
# MTS.hist(methodstrings,ax=ax4,xlabelsize=24,ylabelsize=24,bins=20)
# # ax3.set_xticklabels(methodstrings)
# ax4.set_xlabel("RMSE",fontsize=20)
# ax4.set_ylabel("percent",fontsize=20)
# # ax3.set_xlabel('Methods')
# ax4.grid(True,which="both",axis="y")


# ##
# methodstrings=['fps']
# xpos1=np.arange(len(methodstrings),dtype=np.int)

# fig44, ax44= plt.subplots(figsize=(20, 9))
# FPS=np.array(FPS)
# FPSgg=FPS[FPS<1]
# # ax44.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
# ax44.hist(FPSgg,bins=np.linspace(0,1,30),rwidth=0.9)
# # ax44.set_xticklabels(methodstrings)
# ax44.set_xlabel("Time taken for one scan",fontsize=20)
# ax44.set_ylabel("count",fontsize=20)
# # ax44.set_xlabel('Methods')
# ax44.tick_params(axis='x', labelsize=20 )
# ax44.tick_params(axis='y', labelsize=20 )
# ax44.grid(True,which="both",axis="y")