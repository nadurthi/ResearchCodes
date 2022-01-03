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
from lidarprocessing import point3Dprocessing as pt3dproc
from lidarprocessing import point2Dplotting as pt2dplot
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
from lidarprocessing.numba_codes import gicp
from sklearn.neighbors import KDTree
import os
import pandas as pd
time_increment = 1.736111516947858e-05
angle_increment = 0.004363323096185923
scan_time = 0.02500000037252903
range_min, range_max = 0.023000000044703484, 60.0
angle_min,angle_max =  -2.3518311977386475,2.3518311977386475
from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc,boolean
from numba import njit, prange,jit
import scipy.linalg as sclalg
import scipy.optimize as scopt
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import RotationSpline as RscSpl 
dtype = np.float64
from lidarprocessing import icp
import open3d as o3d
from pykitticustom import odometry
from  uq.filters import pf 

from scipy.interpolate import UnivariateSpline
#%%
basedir =r'P:\SLAMData\Kitti\visualodo\dataset'
# basedir ='/media/na0043/misc/DATA/KITTI/odometry/dataset'
# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
loop_closed_seq = ['02','05','06','08']
sequence = '05'

dataset = odometry.odometry(basedir, sequence, frames=None) # frames=range(0, 20, 5)
Xtpath=np.zeros((len(dataset),6))
f3 = plt.figure()    
ax = f3.add_subplot(111)
for i in range(len(dataset)):
    H=dataset.calib.T_cam0_velo
    H=np.dot(nplinalg.inv(H),dataset.poses[i].dot(H))
    
    Xtpath[i,0:3] = H.dot(np.array([0,0,0,1]))[0:3]
    
    r = Rsc.from_matrix(H[0:3,0:3])
    phidt,xidt,zidt =r.as_euler('zyx', degrees=False)
    Xtpath[i,3:] = phidt,xidt,zidt
    
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

rotations = Rsc.from_euler('zyx', Xtpath[:,3:], degrees=False)
spline =RscSpl(dataset.times, rotations)
AngRates=spline(dataset.times, 1)
AngAcc=spline(dataset.times, 2)

ax.plot(Xtpath[:,0],Xtpath[:,1],'k')
plt.show()

f4 = plt.figure()    
ax = f4.subplots(3,3)
ax[0,0].plot(dataset.times,Xtpath[:,3],'r',label='yaw')
ax[1,0].plot(dataset.times,Xtpath[:,4],'b',label='pitch')
ax[2,0].plot(dataset.times,Xtpath[:,5],'g',label='roll')

ax[0,1].plot(dataset.times,AngRates[:,0],'r',label='Vyaw')
ax[1,1].plot(dataset.times,AngRates[:,1],'b',label='Vpitch')
ax[2,1].plot(dataset.times,AngRates[:,2],'g',label='Vroll')

ax[0,2].plot(dataset.times,AngAcc[:,0],'r',label='Ayaw')
ax[1,2].plot(dataset.times,AngAcc[:,1],'b',label='Apitch')
ax[2,2].plot(dataset.times,AngAcc[:,2],'g',label='Aroll')

ax[0,0].legend()
ax[1,0].legend()
ax[2,0].legend()

plt.show()

f5 = plt.figure()    
ax = f5.subplots(3,2)
ax[0,0].plot(dataset.times,Velocities[:,0],'r',label='vx')
ax[1,0].plot(dataset.times,Velocities[:,1],'r',label='vy')
ax[2,0].plot(dataset.times,Velocities[:,2],'r',label='vz')

ax[0,1].plot(dataset.times,Acc[:,0],'r',label='ax')
ax[1,1].plot(dataset.times,Acc[:,1],'r',label='ay')
ax[2,1].plot(dataset.times,Acc[:,2],'r',label='az')

ax[0,0].legend()
ax[1,0].legend()
ax[2,0].legend()
plt.show()



pose = dataset.poses[1]
velo = dataset.get_velo(2)


#%%

# folder='lidarprocessing/datasets/kitti'

# i=1
# f1="%06d.bin"%i
# X1 = np.fromfile(folder+'/'+f1, dtype=np.float32)
# X1=X1.reshape((-1, 4))
# X1=X1.astype(dtype)

# # pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(X1[:,:3])
# # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)
# # X11=np.asarray(voxel_down_pcd.points)
# # X11=np.ascontiguousarray(X11.T,dtype=dtype)

# X11=X1[1:10000,:3]

# X22=X11+[0,0,1]

# dmax=5
# X11,P1,X22,P2=gicp.gicp_init(X11,X22,dmax=dmax)
# tree1 = KDTree(X11, leaf_size=5)

# x=np.array([0,0,0,0,0,0],dtype=dtype)

# st=time.time()
# res1 = minimize(gicp.gicp_cost, x,args=(tree1,X11,X22,P1,P2,dmax),jac=True,method='BFGS',options={'disp':True,'maxiter':150}) # 'Nelder-Mead'
# et=time.time()
# print("time :",et-st)
# print(res1) 

# gicp.gicp_cost(np.zeros(6),X11,X22,P1,P2,dmax)
#%% PF 
pcd=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)

lines = [[i,i+1] for i in range(Xtpath.shape[0]-1)]
colors = [[0, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(Xtpath[:,:3])
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd,line_set])


pcd_tree = o3d.geometry.KDTreeFlann(pcd)

[k, idx, dd] = pcd_tree.search_hybrid_vector_3d([1,1,1], 2, 2)
#%%





Xlimits=[np.min(np.asarray(pcd.points)[:,0]),np.max(np.asarray(pcd.points)[:,0])]
Ylimits=[np.min(np.asarray(pcd.points)[:,1]),np.max(np.asarray(pcd.points)[:,1])]
Zlimits=[np.min(np.asarray(pcd.points)[:,2]),np.max(np.asarray(pcd.points)[:,2])]

yawlimits=[np.min(Xtpath[:,3]),np.max(Xtpath[:,3])]
pitchlimits=[np.min(Xtpath[:,4]),np.max(Xtpath[:,4])]
rolllimits=[np.min(Xtpath[:,5]),np.max(Xtpath[:,5])]

vv=nplinalg.norm(Velocities,axis=1)
vlimits=[0,np.max(vv)]

omgyawlimits=[np.min(AngRates[:,0]),np.max(AngRates[:,0])]
omgpitchlimits=[np.min(AngRates[:,1]),np.max(AngRates[:,1])]
omgrolllimits=[np.min(AngRates[:,2]),np.max(AngRates[:,2])]

dx=[0.1,0.1,0.1]

Npf=50

def pose2Rt(x):
    t=x[0:3]
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


def dynmodel(xstatepf,dt):
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
        
        Rzphi,dRzdphi=gicp.Rz(phi)
        Ryxi,dRydxi=gicp.Ry(xi)
        Rxzi,dRxdzi=gicp.Rx(zi)
        
        R = Rzphi.dot(Ryxi)
        R=R.dot(Rxzi)
        
        drn = R.dot([1,0,0])
        tdt=t+v*dt*drn
        
        Rzphi,dRzdphi=gicp.Rz(omgyaw*dt)
        Ryxi,dRydxi=gicp.Ry(omgpitch*dt)
        Rxzi,dRxdzi=gicp.Rx(omgroll*dt)
        
        Romg = Rzphi.dot(Ryxi)
        Romg=Romg.dot(Rxzi)
        
        Rdt=R.dot(Romg)
        
        r = Rsc.from_matrix(Rdt)
        phidt,xidt,zidt =r.as_euler('zyx', degrees=False)
        
        xstatepf[i,3]=phidt
        xstatepf[i,4]=xidt
        xstatepf[i,5]=zidt
        
        xstatepf[i,0:3]=tdt
        

    
    return xstatepf
           
# car pose is phi,xi,zi,tpos,v,omgyaw,omgpitch,omgroll    where v is velocity of car, omg is angular velocity

# initialize
xstatepf=np.zeros((Npf,10))


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

wpf=np.ones(Npf)/Npf
dim=xstatepf.shape[1]



robotpf=pf.Particles(X=xstatepf,wts=wpf)

Q=np.diag([(0.25)**2,(0.25)**2,(0.25)**2, # x-y-z
           (10*np.pi/180)**2,(5*np.pi/180)**2,(2*np.pi/180)**2, # angles
           (1)**2,   # velocity
           (0.02*np.pi/180)**2,(0.02*np.pi/180)**2,(0.02*np.pi/180)**2]) # angle rates


figpf = plt.figure()    
axpf = figpf.add_subplot(111)
axpf.plot(Xtpath[:,0],Xtpath[:,1],'k')
axpf.plot(robotpf.X[:,0],robotpf.X[:,1],'r.')
plt.show()  
dmax=1
sig=0.2

#
#
#

XpfTraj=np.zeros((len(dataset),xstatepf.shape[1]))
m,P=robotpf.getEst()
XpfTraj[0]=m

# vis = o3d.visualization.Visualizer()
# vis.create_window()
save_image = False

figpf = plt.figure()    
axpf = figpf.add_subplot(111)

for k in range(1,len(dataset)):
    print(k)
    st=time.time()
    X1v = dataset.get_velo(k)
    et=time.time()
    print("read velo time = ",et-st)
    
    X1v=X1v[:,:3]
    X1v=np.hstack([X1v,np.ones((X1v.shape[0],1))])
    
    H=dataset.calib.T_cam0_velo
    
    # X1=nplinalg.inv(dataset.poses[i]).dot(X1.T).T
    H=np.dot(nplinalg.inv(dataset.calib.T_cam0_velo),dataset.poses[k].dot(H))
    X1gv=H.dot(X1v.T).T
    
    # ff=dataset.calib.T_cam0_velo
    # invff=nplinalg.inv(ff)
    X1gv=X1gv[:,:3]
    
    dt=dataset.times[k]-dataset.times[k-1]
    
    # propagate
    st=time.time()
    robotpf.X=dynmodel(robotpf.X,dt)
    et=time.time()
    print("dyn model time = ",et-st)
    robotpf.X=robotpf.X+np.random.multivariate_normal(np.zeros(dim), Q, Npf)
    
    # measurement update
    # for j in range(Npf):
    #     print("j=",j)
    #     Dv=[]
    #     R,t = pose2Rt(robotpf.X[j][:6])
    #     X1gv_pose = R.dot(X1gv.T).T+t
    #     for i in range(X1gv_pose.shape[0]):
    #         [_, idx, _] = pcd_tree.search_hybrid_vector_3d(X1gv_pose[i], dmax, 1)
    #         if len(idx)>0:
    #             d=nplinalg.norm(pcd.points[idx[0]]-X1gv_pose[i])
    #             Dv.append(d)
    #     if len(Dv)==0:
    #         Dv=[1e5]
    #     likelihood=np.exp(-0.5*np.sum(Dv)/sig**2)
    #     robotpf.wts[j]=likelihood*robotpf.wts[j]
    
    # Dv=Dv.reshape(-1)
    # Xvmidx=Xvmidx.reshape(-1)
    
    # idx=np.argwhere(Dv<dmax)
    # idx=idx.reshape(-1)
    # Dv=Dv[idx]
    
    # Xvmidx=Xvmidx[idx]

    # X11=X1[Xvmidx]
    # X22=X2[idx]
    
    
    # robotpf.renormlizeWts()
    # robotpf.bootstrapResample()
    
    # robotpf.regularizedresample(kernel='gaussian')
    
    m,P=robotpf.getEst()
    XpfTraj[k]=m
    
    # downpcd = pcd.voxel_down_sample(voxel_size=0.1)
    
    # vis.update_geometry(downpcd)
    # vis.poll_events()
    # vis.update_renderer()
    # if save_image:
    #     vis.capture_screen_image("kitti_seq-%s_%06d.jpg" % (sequence,k))
    axpf.cla()
    axpf.plot(Xtpath[:k,0],Xtpath[:k,1],'k')
    axpf.plot(robotpf.X[:,0],robotpf.X[:,1],'r.')
    
    plt.show()
    plt.pause(0.5)
    
# vis.destroy_window()    