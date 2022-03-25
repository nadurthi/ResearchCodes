#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:55:33 2022

@author: na0043
"""

import os
import numpy as np
import yaml
cnt=0
rng = np.random.default_rng()
while 1:
    with open('kitti_localize_config_default.yml', 'r') as outfile:
        D=yaml.safe_load( outfile)
    
    k0 = rng.integers(2000, size=1)
    D['PF']['k0']=k0
    
    with open('kitti_localize_config.yml', 'w') as outfile:
        yaml.dump(D, outfile, default_flow_style=False)
    
    os.system("python /home/na0043/Insync/n.adurthi@gmail.com/Google\ Drive/repos/SLAM/main_kitti_filter_localize_cpp_async_runs.py")
    
    cnt+=1
    
    if cnt==500:
        break