"""Example of pykitti.odometry usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import pykitti

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = os.path.join('P:\\','SLAMData','Kitti','visualodo','dataset')
# basedir = 'P:\SLAMData\\Kitti\visualodo\dataset'

# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
loop_closed_seq = ['02','05','06','08']
sequence = '05'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)
dataset = pykitti.odometry(basedir, sequence, frames=None) # frames=range(0, 20, 5)
Xtpath=np.zeros((len(dataset),4))
f3 = plt.figure()    
ax = f3.add_subplot(111)
for i in range(len(dataset)):
    Xtpath[i,:] = dataset.poses[i].dot(np.array([0,0,0,1]))
ax.plot(Xtpath[:,0],Xtpath[:,2],'k')
plt.show()



mn=np.min(Xtpath,axis=0)
mx=np.max(Xtpath,axis=0)

X=np.zeros((len(dataset),4))
f3 = plt.figure()    
ax = f3.add_subplot(111)
for i in range(len(dataset)):
    X[i,:] = dataset.poses[i].dot(np.array([0,0,0,1]))
    ax.cla()
    ax.plot(X[:i,0],X[:i,2],'k')
    ax.plot(X[i-1,0],X[i-1,2],'ro')
    ax.set_xlim(mn[0]-50,mx[0]+50)
    ax.set_ylim(mn[2]-50,mx[2]+50)
    plt.pause(0.01)

plt.show()

# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of timedelta objects
# dataset.poses:      List of ground truth poses T_w_cam0
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

# Grab some data
second_pose = dataset.poses[1]
first_gray = next(iter(dataset.gray))
first_cam1 = next(iter(dataset.cam1))
first_rgb = dataset.get_rgb(0)
first_cam2 = dataset.get_cam2(0)
third_velo = dataset.get_velo(2)

# Display some of the data
np.set_printoptions(precision=4, suppress=True)
print('\nSequence: ' + str(dataset.sequence))
print('\nFrame range: ' + str(dataset.frames))

print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
print('\nSecond ground truth pose:\n' + str(second_pose))

f, ax = plt.subplots(2, 2, figsize=(15, 5))
ax[0, 0].imshow(first_gray[0], cmap='gray')
ax[0, 0].set_title('Left Gray Image (cam0)')

ax[0, 1].imshow(first_cam1, cmap='gray')
ax[0, 1].set_title('Right Gray Image (cam1)')

ax[1, 0].imshow(first_cam2)
ax[1, 0].set_title('Left RGB Image (cam2)')

ax[1, 1].imshow(first_rgb[1])
ax[1, 1].set_title('Right RGB Image (cam3)')

f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')
# Plot every 100th point so things don't get too bogged down
velo_range = range(0, third_velo.shape[0], 100)
ax2.scatter(third_velo[velo_range, 0],
            third_velo[velo_range, 1],
            third_velo[velo_range, 2],
            c=third_velo[velo_range, 3],
            cmap='gray')
ax2.set_title('Third Velodyne scan (subsampled)')

plt.show()


Xtpath=np.zeros((len(dataset),4))
for i in range(len(dataset)):
    Xtpath[i,:] = dataset.poses[i].dot(np.array([0,0,0,1]))
    
    
f3 = plt.figure()    
ax = f3.add_subplot(111)
ax.plot(Xtpath[:,0],Xtpath[:,2],'k')
plt.show()