# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import os

folder1 = "/home/na0043/Insync/n.adurthi@gmail.com/Google Drive/repos/SLAM/zed_realsense/build/"
folder2= "RealSenseSession_2021-03-24_14-17-39"

folder = folder1+folder2
depthfolder = folder+"/depth"
colorfolder = folder+"/color"

depthfiles_bin = sorted([os.path.join(depthfolder,s) for s in os.listdir(depthfolder) if '.bin' in s])
depthfiles_yml = sorted([os.path.join(depthfolder,s) for s in os.listdir(depthfolder) if '.yml' in s])
colorfiles = sorted([os.path.join(colorfolder,s) for s in os.listdir(colorfolder)])



dfilebin = depthfiles_bin[0]
dfileyml = depthfiles_yml[0]

cfile = colorfiles[0]

fs = cv2.FileStorage(dfileyml, cv2.FILE_STORAGE_READ)
dmatyml = fs.getNode("depth_mat").mat()


[rows,cols]=np.memmap(dfilebin,np.int32,mode='r',offset=0,shape=(2,))
print(rows,cols)
dmat=np.memmap(dfilebin,np.float64,mode='r',offset=8,shape=(rows,cols))
print(dmat.shape)

cmat = cv2.imread(cfile)

cv2.imshow("Color window",cmat)


dmat_norm = np.zeros_like(dmat)
dmat_norm=dmat-np.min(dmat)
dmat_norm = 255*dmat_norm/np.max(dmat_norm)
dmat_norm = dmat_norm.astype(np.int8)
cv2.imshow("Depth window",dmat_norm)