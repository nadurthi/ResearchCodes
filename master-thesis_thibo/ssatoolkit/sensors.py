# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 20:32:21 2022

@author: Nagnanamus
"""

import math
import numpy as np
from ssatoolkit import coords
from ssatoolkit import ukf
from numpy import linalg as nplinalg
from scipy import linalg as sclinalg

def fov(pos, r_site, half_angle, rmax, obs_orientation):
    #range_obs Max range of observation of the observatory
    # theta = obs_pos[0] 
    # phi = obs_pos[1]
    # r = obs_pos[2]
    # r_site = coords.LongLatHeigth_to_cart(obs_pos)
    
    # r_max = np.array([(r + range_obs) * np.cos(theta) * np.cos(phi),
    #                   (r + range_obs) * np.cos(theta) * np.sin(phi),
    #                   (r + range_obs) * np.sin(theta)])
    
    
    
    relpos = pos - r_site
    relpos_unit = relpos/nplinalg.norm(relpos) 
    # print(relpos_unit)
    # print(obs_orientation)
    Beta = np.arccos( np.dot(relpos_unit,obs_orientation) )
    
    # obs_cart = coords.LongLatHeigth_to_cart(obs_pos)
    # pos_to_obs = np.array(
    #     ([pos[0] - obs_cart[0], pos[1] - obs_cart[1], pos[2] - obs_cart[2]]))
    # obs_vect = r_max - obs_cart
    # Beta = math.acos((pos_to_obs.dot(obs_vect)) / (
    #         np.linalg.norm(pos_to_obs) * np.linalg.norm(obs_vect)))
    if abs(Beta) < half_angle and nplinalg.norm(relpos)<=rmax:
        return True
    else:
        return False


def angleNoise2posNoise(Rk):
    aa,wr=ukf.UT_sigmapoints(np.zeros(2), Rk)

    def cc(zk,aa):
        r,theta,phi = coords.cart2spherical(zk[0],zk[1],zk[2])
        # print(anglenoise)
        theta=theta+aa[0]
        phi=phi+aa[1]
        zk = coords.spherical2cart(r,theta,phi)
        return zk
    rr=[]
    for i in range(len(aa)):
        rr.append(cc(7000*np.ones(3),aa[i]))
    rr=np.array(rr)
    a,b = ukf.MeanCov(rr, wr)
    return b



def addNoise(zk,Rk):
    r,theta,phi = coords.cart2spherical(zk[0],zk[1],zk[2])
    anglenoise = np.matmul(sclinalg.sqrtm(Rk) , np.random.randn(2))
    # print(anglenoise)
    theta=theta+anglenoise[0]
    phi=phi+anglenoise[1]
    zk = coords.spherical2cart(r,theta,phi)
    return zk

# x = [pos,vel]
# noise = np.diag([0.01,0.01,0.01]) in radians
class Sensor:
    def __init__(self, idx, Rk, earthpos,  half_angle,rmax, 
                 orientation='normal',plotparams={'color':'r','alpha':0.3, 
                                                  'linewidth':0, 
                                                  'antialiased':False}):
        self.idx = idx
        # self.model= model  # h(x)
        self.earthpos = earthpos
        
        self.half_angle = half_angle
        self.Rk = Rk
        self.Rkpos = angleNoise2posNoise(Rk)
        self.type = 'a'  # 'a' is for angle
        self.rmax = rmax
        self.plotparams=plotparams
        self.r_site = coords.LongLatHeigth_to_cart(earthpos)
        
        if orientation =='normal':
            orientation = self.r_site/nplinalg.norm(self.r_site)
            
        self.orientation = orientation
        
    def inFOV(self,x):
        return fov(x[0:3], self.r_site, self.half_angle,self.rmax, self.orientation)
    
    def evalfunc(self,x):
        # model : z = h(x) + nu
        #zk = self.model(x) + np.matmul(self.noise, np.random.randn(3, 1)) I did not understand what you were trying to do there
        # zk = coords.L_from_cart(x[0:3], self.earthpos)
        zk = coords.L_from_cart_rsite(x[0:3], self.r_site)
        return zk
    
    
    
    
    def gen_meas(self, x):
        # model : z = h(x) + nu
        #zk = self.model(x) + np.matmul(self.noise, np.random.randn(3, 1)) I did not understand what you were trying to do there
        
        if self.inFOV(x) == True:
            # zk = coords.L_from_cart(x[0:3], self.earthpos)
            zk = coords.L_from_cart_rsite(x[0:3], self.r_site)

            # adding angular noise
            r,theta,phi = coords.cart2spherical(zk[0],zk[1],zk[2])
            anglenoise = np.matmul(sclinalg.sqrtm(self.Rk) , np.random.randn(2))
            theta=theta+anglenoise[0]
            phi=phi+anglenoise[1]
            zk = coords.spherical2cart(r,theta,phi)
            

            return zk/nplinalg.norm(zk)
        else:
            return None

     
        
class SensorSet:
    def __init__(self):
        self.sensorList=[]
        
    def append(self,SS):
        self.sensorList.append(SS)
    
    def __getitem__(self,idx):
        for i in range(len(self.sensorList)):
            if self.sensorList[i].idx==idx:
                return self.sensorList[i]
    def __len__(self):
        return len(self.sensorList)
    
    def getsensorIDs(self):
        ids = sorted([ss.idx for ss in self.sensorList])
        return ids        
    def itersensors(self):
        ids = self.getsensorIDs()
        for idx in ids:
            yield self.__getitem__(idx)
            
            