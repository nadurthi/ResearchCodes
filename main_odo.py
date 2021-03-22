#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:14:18 2020

@author: nagnanamus
"""
import time
from matplotlib import pyplot as plt
from visionprocessing import drawing as vpdrw
from visionprocessing import stereo as vpst
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')


img0 = cv2.imread('lefttest.png', 3)
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

img1 = cv2.imread('righttest.png', 3)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)


gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

st = time.time()
featpairs, matches, ptpairs = vpst.get_matched_features_imgpairs(gray0, gray1, featdes0=None,
                                                                 typefeat='orb',
                                                                 method='NewPts',
                                                                 matchmethod='BFknn',
                                                                 maxfeats=100)
print(time.time() - st)

st = time.time()
lkpairs = vpst.get_matched_features_LK_imgpairs(gray0, gray1,
                                                feat0=None,
                                                typefeat='orb')
print(time.time() - st)


img3 = vpdrw.drawMatchedPtsImgpairs(
    img0, img1, lkpairs.p0[:20, :], lkpairs.p1[:20, :])
fig = plt.figure()
plt.imshow(img3, aspect='auto')
plt.autoscale(False)
plt.show()


img4 = vpdrw.drawMatchedPtsImgpairs(
    img0, img1, ptpairs.p0[:20, :], ptpairs.p1[:20, :])
fig = plt.figure()
plt.imshow(img4, aspect='auto')
plt.autoscale(False)
plt.show()

f = 7.188560000000e+02
b_gray = 0.5371657081270053
b_rgb = 0.5323318472000029
Kl = np.array([[718.856, 0., 607.1928],
               [0., 718.856, 185.2157],
               [0., 0., 1.]])
Kr = np.array([[718.856, 0., 607.1928],
               [0., 718.856, 185.2157],
               [0., 0., 1.]])
pt3d = vpst.compute_3D_points_stereopair(
    ptpairs.p0, ptpairs.p1, Kl, Kr, f, b_rgb)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pt3d[:, 0], pt3d[:, 1], pt3d[:, 2], marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# %%
