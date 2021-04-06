# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import os
import pandas as pd
import pyrealsense2 as rs
import open3d as o3d
from pypcd import pypcd
# from pypcd import PointCloud
import pprint
import matplotlib.pyplot as plt

mainfolder = "/media/na0043/misc/DATA/MyZEDL515"
realfolder= os.path.join(mainfolder,  "RealSenseSession_2021-04-06_10-23-23")
zedfolder= os.path.join(mainfolder,  "ZedSession_2021-04-06_13-20-20")

#%% real sense

# depth is aligned to color image
depthfolder = realfolder+"/depth"
colorfolder = realfolder+"/color"
pcdfolder = realfolder+"/pcd"
times = pd.read_csv(realfolder+'/timesteps.csv',header=None,names=['T'])




depthfiletag = 'depthCV_64F_'
colorfiletag = 'color_'
pcdfiletag = 'pcd_'

depthfiles_bin = sorted([os.path.join(depthfolder,s) for s in os.listdir(depthfolder) if '.bin' in s])
depthfiles_yml = sorted([os.path.join(depthfolder,s) for s in os.listdir(depthfolder) if '.yml' in s])
colorfiles = sorted([os.path.join(colorfolder,s) for s in os.listdir(colorfolder)])
pcdfiles = sorted([os.path.join(pcdfolder,s) for s in os.listdir(pcdfolder)])

N=len(depthfiles_bin)



dfilebin = depthfiles_bin[100]


cfile = colorfiles[100]

# dfileyml = depthfiles_yml[0]
# fs = cv2.FileStorage(dfileyml, cv2.FILE_STORAGE_READ)
# dmatyml = fs.getNode("depth_mat").mat()


[rows,cols,elmsize]=np.memmap(dfilebin,np.int32,mode='r',offset=0,shape=(3,))
print(rows,cols)
dtype = None
if elmsize == 8:
    dtype = np.float64
elif elmsize == 4:
    dtype = np.float32
    
dmat=np.memmap(dfilebin,dtype,mode='r',offset=3*elmsize,shape=(rows,cols))
print(dmat.shape)

# color
cmat = cv2.imread(cfile)
cv2.imshow("Color window",cmat)

colorintrinsics = rs.intrinsics()
colorintrinsics.coeffs = [0.181048,-0.564822,-0.0010648,5.49983e-05,0.508008]
colorintrinsics.fx = 609.292
colorintrinsics.fy = 609.546
colorintrinsics.height = 480
colorintrinsics.model = rs.distortion(2) #"Inverse Brown Conrady"
colorintrinsics.ppx = 317.698
colorintrinsics.ppy = 247.739
colorintrinsics.width = 640

depthintrinsics = rs.intrinsics()
depthintrinsics.coeffs = [0,0,0,0,0]
depthintrinsics.fx = 458.457
depthintrinsics.fy = 458.438
depthintrinsics.height = 480
depthintrinsics.model = rs.distortion(0) #"Inverse Brown Conrady"
depthintrinsics.ppx = 339.879
depthintrinsics.ppy = 247.33
depthintrinsics.width = 640

depth_to_color_extrin = rs.extrinsics()
depth_to_color_extrin.rotation = [0.999983,-0.00571586,0.00109741,0.00574274,0.999638,-0.0262948,-0.000946714,0.0263006,0.999654] # col major
# depth_to_color_extrin.rotation = [0.999983,0.00574274,-0.000946714,-0.00571586,0.999638,0.0263006,0.00109741,-0.0262948,0.999654] # row-major
depth_to_color_extrin.translation = [-0.000311315,0.0138645,-0.00281649]
# pcd

cloud = pypcd.PointCloud.from_path(pcdfiles[100])
pprint.pprint(cloud.get_metadata())

pcd = o3d.geometry.PointCloud()
xyz = np.vstack([list(s) for s in cloud.pc_data])
pcd.points = o3d.utility.Vector3dVector(xyz) 
np_colors = np.array([(0.8,0.1,0.1) for i in range(xyz.shape[0])]) 
pcd.colors = o3d.utility.Vector3dVector(np_colors)
o3d.visualization.draw_geometries([pcd])


#%% projection
hp=241 # height
wp=321 # width

a = rs.rs2_deproject_pixel_to_point(colorintrinsics, [wp,hp], dmat[hp,wp])
print(a,cloud.pc_data[(hp)*640+wp])

cnt=0
xyz_rgbd = np.zeros((cmat.shape[0]*cmat.shape[1],3))
for pw in range(cmat.shape[1]):
    for ph in range(cmat.shape[0]):
        a = rs.rs2_deproject_pixel_to_point(colorintrinsics, [pw,ph], dmat[ph,pw])
        xyz_rgbd[cnt,:] = a #[a[1],a[0],a[2]]
        
        # depthpoint  = rs.rs2_deproject_pixel_to_point(depthintrinsics, [i,j], dmat[i,j])
        # colorpoint = rs.rs2_transform_point_to_point(depth_to_color_extrin,depthpoint)
        # xyz_rgbd[cnt,:] = [colorpoint[1],colorpoint[0],colorpoint[2]]
        
        cnt+=1

d=xyz_rgbd[:480,2]-xyz[:480,2]
print([min(d),max(d)])

pcd2 = o3d.geometry.PointCloud()        
pcd2.points = o3d.utility.Vector3dVector(xyz_rgbd)
np_colors = np.array([(0.1,0.8,0.1) for i in range(xyz_rgbd.shape[0])]) 
pcd2.colors = o3d.utility.Vector3dVector(np_colors)
o3d.visualization.draw_geometries([pcd,pcd2])


#%% -----------------
# --------------------------------------------------------------------------

#%% ----------------------------------------------------

#%% zed camera


depthfolder = zedfolder+"/depth"
colorfolder = zedfolder+"/color"
pcdfolder = zedfolder+"/pcd"
times = pd.read_csv(zedfolder+'/timesteps.csv',header=None,names=['T'])




depthfiletag = 'depth_'
colorfiletag = 'color_'
pcdfiletag = 'pcd_'

depthfiles_bin = sorted([os.path.join(depthfolder,s) for s in os.listdir(depthfolder) if '.bin' in s])
depthfiles_yml = sorted([os.path.join(depthfolder,s) for s in os.listdir(depthfolder) if '.yml' in s])
colorfiles_bin = sorted([os.path.join(colorfolder,s) for s in os.listdir(colorfolder) if '.bin' in s])
colorfiles_png = sorted([os.path.join(colorfolder,s) for s in os.listdir(colorfolder) if '.png' in s])
pcdfiles = sorted([os.path.join(pcdfolder,s) for s in os.listdir(pcdfolder)])

N=len(depthfiles_bin)



dfilebin = depthfiles_bin[100]

cpngf = colorfiles_png[0]

cmat = cv2.imread(cpngf)
cv2.imshow("Color window",cmat)


# read depth
[rows,cols,elmsize]=np.memmap(dfilebin,np.int32,mode='r',offset=0,shape=(3,))
print(rows,cols,elmsize)
dtype = None
if elmsize == 8:
    dtype = np.float64
elif elmsize == 4:
    dtype = np.float32
    
dmat=np.memmap(dfilebin,dtype,mode='r',offset=3*elmsize,shape=(rows,cols))
print(dmat.shape)


# point cloud
cloud = pypcd.PointCloud.from_path(pcdfiles[100])
pprint.pprint(cloud.get_metadata())

pcd = o3d.geometry.PointCloud()
xyz = np.vstack([list(s)[0:3] for s in cloud.pc_data])
pcd.points = o3d.utility.Vector3dVector(xyz) 
np_colors = np.array([list(s)[3] for s in cloud.pc_data],dtype=np.float32) 
Mat mResult;
mFloatFrame.convertTo(mResult, CV_8UC4);
# pcd.colors = o3d.utility.Vector3dVector(np_colors)
o3d.visualization.draw_geometries([pcd])
