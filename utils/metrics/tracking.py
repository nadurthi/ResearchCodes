# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import numpy.linalg as nplinalg

def multiTarget_tracking_metrics(tf,groundtargetset,targetset):
    # tracking error for active times
    ErrPosTrack={}
    ErrVelTrack={}
    ActiveTimeSteps={}
    TimeStepsBeforeFirstTracked={}
    MaxTimeInActive={}
    CountGapsInactive={}
    
    DF=[]
    
    for i in range(targetset.ntargs):
        if targetset[i].isSearchTarget():
            continue
        tu,xuk = targetset[i].recorderpost.getvar_uptotime_list('xfk',tf,returntimes=True)
        tu,status = targetset[i].recorderpost.getvar_uptotime_list('status',tf,returntimes=True)
        
        targID = targetset[i].ID
        
        gndtarget = groundtargetset.getTargetByID(targID)
        tt,xtk = gndtarget.groundtruthrecorder.getvar_uptotime_list('xtk',tf,returntimes=True)
        
        df=pd.DataFrame(columns=['status'],index=tt)
        df['status']='InActive'
        
        for i in len(tu):
            df.loc[tu[i],'status'] = status[i]
            
        DF[targetset[i].targetName] = df
        
        dfact = df[df['status']=='Active']
        epos=0
        evel=0
        for t in dfact.index:
            it = tt.index(t)
            iu = tu.index(t)
            e = xtk[it]-xuk[iu]
            epos = epos + nplinalg.norm(e[0:2])
            evel = evel + nplinalg.norm(e[2:4])
            
        # get metrics         
        ErrPosTrack[targetset[i].targetName] = epos
        ErrVelTrack[targetset[i].targetName] = evel
        
        
                