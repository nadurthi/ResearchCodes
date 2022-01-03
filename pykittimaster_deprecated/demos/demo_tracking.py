"""Example of pykitti.odometry usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import pykitti.tracking2 as pyktrack
import pdb

plt.close("all")
# Change this to the directory where you store KITTI data
# basedir = os.path.join('P:\\','SLAMData','Kitti','tracking','mot','training')
# basedir = 'P:\SLAMData\\Kitti\visualodo\dataset'
basedir = '/media/na0043/misc/DATA/KITTI/mots/dataset/training'

# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
# loop_closed_seq = ['02','05','06','08']
# sequence = '05'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)

# basedir = os.path.join('P:','SLAMData','Kitti','tracking','mot','training')
# seq = ['0007','0009','0019','0020'];
seq = '0000';
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
plt.close("all")


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
    plt.figure(1)
    ax.imshow(imgL)
    xlm=ax.get_xlim()
    ylm=ax.get_ylim()
    
    
    tracklets = dflabel[dflabel['frame']==i]
    print(tracklets[['classtype','roty','alpha','beta','beta2','tx','ty','tz']])
    for ind in tracklets.index:
        plt.figure(1)
        pyktrack.drawBox2D(ax,tracklets.loc[ind,:] )
        corners,face_idx = pyktrack.computeBox3D(tracklets.loc[ind,:],ktrk.calib.P_rect_20)
        # if len(corners)>0:
        #     if np.max(np.abs(np.array(corners).reshape(-1)))>1000:
        #         pdb.set_trace()
            
        orientation = pyktrack.computeOrientation3D(tracklets.loc[ind,:],ktrk.calib.P_rect_20)
        pyktrack.drawBox3D(ax, tracklets.loc[ind,:],corners,face_idx,orientation)
        ax.set_title("frame id = "+str(i))
        ax.set_xlim(xlm)
        ax.set_ylim(ylm)
        
        fig.canvas.draw()
        
        if tracklets.loc[ind,'classtype'] != 'DontCare':
            plt.figure(2)
            ax2.plot([0,tracklets.loc[ind,'tx']],[0,tracklets.loc[ind,'tz']])
            fig2.canvas.draw()
    
    
    oxts = ktrk.get_oxts(i)
    # timu = np.matmul(oxts.T_w_imu,np.array([0,0,0,1]) )
    timu = oxts.T_w_imu.dot(np.array([0,0,0,1]))
    Ximu[i,:] = timu[:3]
    plt.figure(3)
    ax3.plot(-Ximu[:i,1],Ximu[:i,0])
    ax3.plot(-Ximu[i-1,1],Ximu[i-1,0],'ro')
    fig3.canvas.draw()
    
    plt.pause(0.2)
    plt.show()
    
    
#    % plot 3D bounding box

#    img = np.hstack([imgL,imgR])
#    img_rgb = np.hstack([imgL_rgb,imgR_rgb])
#    ax.imshow(img_rgb)
    plt.show()
    plt.pause(1)
    # plt.waitforbuttonpress()

#    break


fig3= plt.figure(4)
ax3 = fig3.add_subplot(111)
ax3.plot(Ximu[:i,0],Ximu[:i,1])
ax3.plot(Ximu[i-1,0],Ximu[i-1,1],'ro')
    