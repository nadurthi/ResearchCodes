# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import uq.quadratures.cubatures as uqcb
from loggerconfig import *
import numpy as np
import numpy.linalg as nplg
from scipy.linalg import block_diag
import collections as clc
import pandas as pd
import os
import dill,pickle
import cv2
import pykitticustom
import pykitticustom as pykcus
from pykitticustom import tracking2 as pykcustracking
import time
import cv2
import datetime as dt
import glob
import os
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('display.max_columns', None)
#%%
logger = logging.getLogger(__name__)

logger.info('Info log message')
logger.debug('debug message')

logger.error('error example')
logger.verbose('verbose log message')
# try:
#     raise Exception('exception message')
# except:
#     logger.exception('error occured')


logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
# %%

from uq.motfilter import mot as uqmot
from uq.motfilter import jpda
from uq.filters.kalmanfilter import TargetKF
from uq.motfilter import targets as uqtargets
from uq.uqutils.random import genRandomMeanCov
from physmodels.motionmodels import KinematicModel_UM
from physmodels.sensormodels import DiscLTSensorModel
import uq.motfilter.measurements as motmeas
from uq.uqutils import recorder as uqrecorder
from uq.uqutils import helper as uqutilhelp
from uq.uqutils import metrics as uqmetrics
from uq.uqutils import simmanager as uqsimmanager

# if __name__=='__main__':

#%% testing kitti read

root_dir = '/media/nagnanamus/d0690b96-7f71-44f2-96da-9f7259180ec7/SLAMData/Kitti/tracking/mot/allcomb/training';
seq = '0000';
ktrk = pykcustracking.KittiTracking(root_dir,seq)
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


#%%



rad2deg = 180/np.pi
fig= plt.figure(1)
ax = fig.add_subplot(111)

fig2= plt.figure(2)
ax2 = fig2.add_subplot(111)

fig3= plt.figure(3)
ax3 = fig3.add_subplot(111)
Ximu=np.zeros((200,3))

for i in range(ktrk.nframes):
    ax.cla()
    ax2.cla()
    ax3.cla()

    imgL = ktrk.get_cam2(i)
    imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    imgR = ktrk.get_cam3(i)
    imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    ax.imshow(imgL_rgb)

    tracklets = dflabel[dflabel['frame']==i]
    print(tracklets[['classtype','roty','alpha','beta','beta2','tx','ty','tz']])
    for ind in tracklets.index:
        pykcustracking.drawBox2D(ax,tracklets.loc[ind,:] )
        corners,face_idx = pykcustracking.computeBox3D(tracklets.loc[ind,:],ktrk.calib.P_rect_20)
        orientation = pykcustracking.computeOrientation3D(tracklets.loc[ind,:],ktrk.calib.P_rect_20)
        pykcustracking.drawBox3D(ax, tracklets.loc[ind,:],corners,face_idx,orientation)


        if tracklets.loc[ind,'classtype'] != 'DontCare':
            ax2.plot([0,tracklets.loc[ind,'tx']],[0,tracklets.loc[ind,'tz']])

    oxts = ktrk.get_oxts(i)
#    timu = np.matmul(nplg.inv(oxts.T_w_imu),np.array([0,0,0,1]) )
    timu = np.matmul(oxts.T_w_imu,np.array([0,0,0,1]) )
#    Ximu[i,:] = np.matmul(np.array([[0,-1,0],[1,0,0],[0,0,1]]),timu[:3])
    Ximu[i,:] = timu[:3]
    ax3.plot(Ximu[:i,0],Ximu[:i,1])
    ax3.plot(Ximu[i-1,0],Ximu[i-1,1],'ro')

#    % plot 3D bounding box

#    img = np.hstack([imgL,imgR])
#    img_rgb = np.hstack([imgL_rgb,imgR_rgb])
#    ax.imshow(img_rgb)
    plt.show()
#    plt.pause(0.2)
    plt.waitforbuttonpress()

#    break


