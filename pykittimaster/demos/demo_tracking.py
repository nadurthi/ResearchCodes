"""Example of pykitti.odometry usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import pykitti.tracking2 as pyktrack

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = os.path.join('P:\\','SLAMData','Kitti','tracking','mot','training')
# basedir = 'P:\SLAMData\\Kitti\visualodo\dataset'

# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
# loop_closed_seq = ['02','05','06','08']
# sequence = '05'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)

basedir = os.path.join('P:','SLAMData','Kitti','tracking','mot','training')
# seq = ['0007','0009','0019','0020'];
seq = '0009';
ktrk = pyktrack.KittiTracking(basedir,seq)
dflabel = ktrk.readlabel()
dflabel['beta'] = (dflabel['roty']-dflabel['alpha'])*180/np.pi
dflabel['beta2'] = np.arctan(dflabel['tx']/dflabel['tz'])*180/np.pi
print(ktrk.classlabels)

#    cv2.imshow('Window', img)
#
#    key = cv2.waitKey(1000)#pauses for 3 seconds before fetching next image
#    if key == 27:#if ESC is pressed, exit loop
##        cv2.destroyAllWindows()
#        break

#plt.close('all')
#cv2.destroyAllWindows()

print(ktrk.nframes)
#%%



rad2deg = 180/np.pi
fig= plt.figure(1)
ax = fig.add_subplot(111)

fig2= plt.figure(2)
ax2 = fig2.add_subplot(111)

fig3= plt.figure(3)
ax3 = fig3.add_subplot(111)
Ximu=np.zeros((ktrk.nframes,3))

for i in range(ktrk.nframes):
    ax.cla()
    ax2.cla()
    ax3.cla()

    imgL = ktrk.get_cam2(i)
    # imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    # imgR = ktrk.get_cam3(i)
    # imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    ax.imshow(imgL)
    
    
    tracklets = dflabel[dflabel['frame']==i]
    print(tracklets[['classtype','roty','alpha','beta','beta2','tx','ty','tz']])
    for ind in tracklets.index:
        pyktrack.drawBox2D(ax,tracklets.loc[ind,:] )
        corners,face_idx = pyktrack.computeBox3D(tracklets.loc[ind,:],ktrk.calib.P_rect_20)
        orientation = pyktrack.computeOrientation3D(tracklets.loc[ind,:],ktrk.calib.P_rect_20)
        pyktrack.drawBox3D(ax, tracklets.loc[ind,:],corners,face_idx,orientation)
    

        if tracklets.loc[ind,'classtype'] != 'DontCare':
            ax2.plot([0,tracklets.loc[ind,'tx']],[0,tracklets.loc[ind,'tz']])
    
    plt.show()
    plt.pause(1)
    
    oxts = ktrk.get_oxts(i)
    # timu = np.matmul(oxts.T_w_imu,np.array([0,0,0,1]) )
    timu = oxts.T_w_imu.dot(np.array([0,0,0,1]))
    Ximu[i,:] = timu[:3]
    ax3.plot(Ximu[:i,0],Ximu[:i,1])
    ax3.plot(Ximu[i-1,0],Ximu[i-1,1],'ro')

#    % plot 3D bounding box

#    img = np.hstack([imgL,imgR])
#    img_rgb = np.hstack([imgL_rgb,imgR_rgb])
#    ax.imshow(img_rgb)
    plt.show()
    plt.pause(0.2)
    # plt.waitforbuttonpress()

#    break


fig3= plt.figure(4)
ax3 = fig3.add_subplot(111)
ax3.plot(Ximu[:i,0],Ximu[:i,1])
ax3.plot(Ximu[i-1,0],Ximu[i-1,1],'ro')
    