# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:56:41 2020

@author: Nagnanamus
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

filepath = "../houseScan.pkl"
with open(filepath,'rb') as fh:
    ptsdata = pkl.load(fh)
            
for idx in range(5):
    fig,ax = plt.subplots(1, 1)

    rngs = np.array(ptsdata[idx]['ranges'])
            
    angle_min=ptsdata[idx]['angle_min']
    angle_max=ptsdata[idx]['angle_max']
    angle_increment=ptsdata[idx]['angle_increment']
    ths = np.arange(angle_min,angle_max+angle_increment,angle_increment)
    p=np.vstack([np.cos(ths),np.sin(ths)])
    
    
    ptset = rngs.reshape(-1,1)*p.T
    
    safeptsidx = (rngs<=ptsdata[idx]['range_max']) & (rngs>=ptsdata[idx]['range_min'])
    ptset = ptset[safeptsidx,:]
    
    ax.plot(ptset[:,0],ptset[:,1],'bo')
