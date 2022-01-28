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
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import RotationSpline as RscSpl 
from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point3Dprocessing as pt3dproc
from lidarprocessing import point2Dplotting as pt2dplot
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
import lidarprocessing.numba_codes.binmatchers as binmatchers
from lidarprocessing.numba_codes import gicp
from scipy.interpolate import UnivariateSpline
dtype = np.float64
import open3d as o3d
from pykitticustom import odometry
from  uq.filters import pf 
import time
import pickle
from joblib import dump, load
from pyslam import  slam
import json

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




#%% PF 


pcd3D=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)
pcd3DdownSensorCost=pcd3D.voxel_down_sample(voxel_size=0.5)


pcd3Droadremove=o3d.io.read_point_cloud("kitti-pcd-seq-roadremove-%s.pcd"%sequence)
np.asarray(pcd3Droadremove.points)[:,2]=0
X2Dmap_down=np.array(pcd3Droadremove.voxel_down_sample(voxel_size=0.1).points)
X2Dmap_down=X2Dmap_down[:,:2]


D={"icp":{},
   "gicp":{},
   "gicp_cost":{},
   "ndt":{},
   "sig0":0.5,
   "dmax":10,
   "DoIcpCorrection":0}

D["icp"]["enable"]=1
D["icp"]["setMaximumIterations"]=50
D["icp"]["setMaxCorrespondenceDistance"]=25
D["icp"]["setRANSACIterations"]=0.0
D["icp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["icp"]["setTransformationEpsilon"]=1e-3
D["icp"]["setEuclideanFitnessEpsilon"]=1



D["gicp_cost"]["enable"]=0


D["gicp"]["enable"]=0
D["gicp"]["setMaxCorrespondenceDistance"]=20
D["gicp"]["setMaximumIterations"]=50.0
D["gicp"]["setMaximumOptimizerIterations"]=20.0
D["gicp"]["setRANSACIterations"]=0
D["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["gicp"]["setTransformationEpsilon"]=1e-9
D["gicp"]["setUseReciprocalCorrespondences"]=1

D["ndt"]["enable"]=0
D["ndt"]["setTransformationEpsilon"]=1e-9
D["ndt"]["setStepSize"]=2.0
D["ndt"]["setResolution"]=1.0
D["ndt"]["setMaximumIterations"]=25.0
D["ndt"]["initialguess_axisangleA"]=0.0
D["ndt"]["initialguess_axisangleX"]=0.0
D["ndt"]["initialguess_axisangleY"]=0.0
D["ndt"]["initialguess_axisangleZ"]=1.0
D["ndt"]["initialguess_transX"]=0.5
D["ndt"]["initialguess_transY"]=0.01
D["ndt"]["initialguess_transZ"]=0.01

D["DON"]={}
D["DON"]["scale1"]=1;
D["DON"]["scale2"]=2;
D["DON"]["threshold"]=0.2;
D["DON"]["threshold_small_z"]=0.5;
D["DON"]["threshold_large_z"]=0.5;

D["DON"]["segradius"]=1;

D["DON"]["threshold_curv_lb"]=0.1;
D["DON"]["threshold_curv_ub"]=100000;
D["DON"]["threshold_small_nz_lb"]=-0.5;
D["DON"]["threshold_small_nz_ub"]=0.5;
D["DON"]["threshold_large_nz_lb"]=-5;
D["DON"]["threshold_large_nz_ub"]=5;

pclloc=slam.Localize(json.dumps(D))

pcd3DdownSensorCostDown=pcd3DdownSensorCost.voxel_down_sample(voxel_size=0.25)
X3Dmap_down = np.asarray(pcd3DdownSensorCostDown.points)
pclloc.setMapX(X3Dmap_down)

opts = json.dumps(D)





#%%
Npf = 100

def down_sample(X,voxel_size):
    X=np.asarray(X)
    NN=X.ndim
    if NN==2:
        X=np.hstack([X,np.zeros((X.shape[0],1))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    XX=np.asarray(voxel_down_pcd.points)
    return XX[:,:NN]





Xlimits_scan=[np.min(np.asarray(pcd3DdownSensorCost.points)[:,0]),np.max(np.asarray(pcd3DdownSensorCost.points)[:,0])]
Ylimits_scan=[np.min(np.asarray(pcd3DdownSensorCost.points)[:,1]),np.max(np.asarray(pcd3DdownSensorCost.points)[:,1])]
Zlimits_scan=[np.min(np.asarray(pcd3DdownSensorCost.points)[:,2]),np.max(np.asarray(pcd3DdownSensorCost.points)[:,2])]


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
    
    Q=np.diag([(0.1)**2,(0.1)**2,(0.1)**2, # x-y-z
                (2*np.pi/180)**2,(2*np.pi/180)**2,(0.1*np.pi/180)**2, # angles
                (0.2)**2,   # velocity
                (0.05*np.pi/180)**2,(0.05*np.pi/180)**2,(0.02*np.pi/180)**2,# angle rates
                (0.01)**2]) #acc
    
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



Rposnoise=np.diag([(1)**2,(1)**2,(1)**2])
    
sampligfunc = getCTinitialSamples
# sampligfunc= getCTinitialSamples_origin

xstatepf, wpf, Q =sampligfunc(Npf)
# xstatepf, wpf, Q =getUMinitialSamples(Npf)

dim=xstatepf.shape[1]
robotpf=pf.Particles(X=xstatepf,wts=wpf)


dmax=D["dmax"]

XpfTraj=np.zeros((len(dataset),xstatepf.shape[1]))
m,P=robotpf.getEst()
XpfTraj[0]=m



figpf = plt.figure()    
axpf = figpf.add_subplot(111,projection='3d')

    
for k in range(1,len(dataset)):
    print(k)
    st=time.time()
    X1v = dataset.get_velo(k)
    et=time.time()
    print("read velo time = ",et-st)
    
    X1v=X1v[:,:3]
    
    idxx=(X1v[:,0]>-200) & (X1v[:,0]<200) & (X1v[:,1]>-200) & (X1v[:,1]<200 )& (X1v[:,2]>-100) & (X1v[:,2]<100)
    X1v=X1v[idxx,:]

    Hgt=dataset.calib.T_cam0_velo
    
    
    Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[k].dot(Hgt))
    X1gv=Hgt[0:3,0:3].dot(X1v.T).T+Hgt[0:3,3]

    dt=dataset.times[k]-dataset.times[k-1]
    
    # propagate
    st=time.time()
    # robotpf.X=dynmodelUM3D(robotpf.X,dt)
    robotpf.X=dynmodelCT3D(robotpf.X,dt)
    
    et=time.time()
    print("dyn model time = ",et-st)
    robotpf.X=robotpf.X+2*np.random.multivariate_normal(np.zeros(dim), Q, Npf)
    
    # measurement update
    
    LiK=[]
    
    pcdXv = o3d.geometry.PointCloud()
    pcdXv.points = o3d.utility.Vector3dVector(X1v[:,:3])
    pcdXv_down_pcd = pcdXv.voxel_down_sample(voxel_size=0.5)
    X1v_down = np.asarray(pcdXv_down_pcd.points)
    
    
    #
    idxpf=0
    doBinMatch=0
    if doBinMatch:
        st=time.time()
        ret=pclloc.computeLikelihood(robotpf.X,X1v_down)
        ret=dict(ret)
        beforelikelihood=ret['likelihood'].reshape(-1)
        
  
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
        
        dxMatch=np.array([2,2])
        dxBase=np.array([30,30])
        Lmax=np.array([200,200])
        thmax=170*np.pi/180
        thmin=2.5*np.pi/180
        
        idxpfmx=np.argmax(robotpf.wts)
        for idxpf in [idxpfmx]:
            
            Ridx,tidx = pose2Rt(robotpf.X[idxpf][:6])
            phiidx = robotpf.X[idxpf][3]
            Rzphiidx,_=gicp.Rz(phiidx)
            
            H12est=np.identity(3)
            H12est[0:2,0:2]=Rzphiidx[0:2,0:2]
            H12est[0:2,2]=tidx[:2]
            
            X1v2D=X1v_roadrem[:,:2].copy()
            X1v2D=down_sample(X1v2D,dxMatch[0])
            #ax.cla()
            #ax.plot(X1v_roadrem[:,0],X1v_roadrem[:,1],'k.')
        
            X1v2Dgpf=H12est[0:2,0:2].dot(X1v2D.T).T+H12est[0:2,2]
        
        
            X1gvdown=down_sample(X1gv[:,:2],dxMatch[0])
        
        
            st=time.time()
            # Hbin21=binMatcherAdaptive3(X11,X2,H12est,Lmax,thmax,thmin,dxMatch)
            # Hbin21,costs=binMatcherAdaptive3(X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch)
            Hbin21,cost0,cost,hh,hR=binmatchers.binMatcherAdaptive_super(X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase)
            et=time.time()
            
            Hbin12 = nplinalg.inv(Hbin21)
            X1v2Dgc=Hbin12[0:2,0:2].dot(X1v2D.T).T+Hbin12[0:2,2]
            
            figbf = plt.figure("bin-fit")
            if len(figbf.axes)>0:
                ax = figbf.axes[0]
            else:
                ax = figbf.add_subplot(111)
            ax.cla()
            ax.plot(X2Dmap_down[:,0],X2Dmap_down[:,1],'k.')
            ax.plot(X1v2Dgpf[:,0],X1v2Dgpf[:,1],'b.',label='pf-pose')
            ax.plot(X1v2Dgc[:,0],X1v2Dgc[:,1],'r.',label='pf-pose-corrected')
            ax.plot(X1gv_roadrem[:,0],X1gv_roadrem[:,1],'g.',label='ground truth')
            ax.legend()
            ax.axis("equal")
            
            
            tc,thc=nbpt2Dproc.extractPosAngle(Hbin12)
            
            
            if cost>cost0:
                # idxpfleast=np.argmin(robotpf.wts)
                # robotpf.X[idxpfleast]=robotpf.X[idxpf]
                # robotpf.wts[idxpfleast]=robotpf.wts[idxpf]
                # robotpf.X[idxpfleas50t][3]=thc
                # robotpf.X[idxpfleast][:2]=tc
                
                robotpf.X[idxpf][3]=thc
                robotpf.X[idxpf][:2]=tc
                # robotpf.wts[idxpf]=0.8
                # print(thc)
                Ridx,tidx = pose2Rt(robotpf.X[idxpf][:6])
                Hidx=np.identity(4)
                Hidx[0:3,0:3]=Ridx
                Hidx[0:3,3]=tidx
                X1v_down_pfpose = Ridx.dot(X1v_down.T).T+tidx
                
                Hpcl={'H_icp':np.identity(4)}
                # bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(X1v_down_pfpose,axis=0)-5,max_bound=np.max(X1v_down_pfpose,axis=0)+5)
                # pcdbox=pcd3DdownSensorCostDown.crop(bbox3d)
                # Xmappflocal = np.asarray(pcdbox.points)
                # Hpcl=slam.registrations(Xmappflocal.copy(),X1v_down_pfpose.copy(),json.dumps(D))
                # Hpcl=dict(Hpcl)
                Ha=Hpcl['H_icp'].dot(Hidx)
                xx=Rt2pose(Ha[0:3,0:3],Ha[0:3,3])
                robotpf.X[idxpf][:6]=xx
                pass
            
        robotpf.renormlizeWts()
    
    print("Pf#%d wt before = ",robotpf.wts[idxpf])
    st=time.time()
    ret=pclloc.computeLikelihood(robotpf.X,X1v_down)
    ret=dict(ret)
    likelihood=ret['likelihood'].reshape(-1)

    likelihood=np.exp(-likelihood)
    likelihood[likelihood<1e-20]=1e-20
    # break
    robotpf.wts=likelihood*robotpf.wts
    et=time.time()
    print("pcl meas model time = ",et-st)
    
    robotpf.renormlizeWts()
    print("Pf#%d wt after = ",robotpf.wts[idxpf])
    
   
    
    Neff=robotpf.getNeff()
    if Neff/Npf<0.25:
        robotpf.bootstrapResample()
    
    # robotpf.regularizedresample(kernel='gaussian',scale=1/25)
    
    m,P=robotpf.getEst()
    XpfTraj[k]=m
    u,v=nplinalg.eig(P)
    

    

    axpf.cla()
    axpf.plot(Xtpath[:k,0],Xtpath[:k,1],Xtpath[:k,2],'k')
    axpf.plot(robotpf.X[:,0],robotpf.X[:,1],robotpf.X[:,2],'r.')
        
    # xmap = np.asarray(pcdbox.points)
    # axpf.plot(xmap[:,0],xmap[:,1],xmap[:,2],'k.')
    
    
    plt.pause(0.1)
    
    if k>=400:
        break
