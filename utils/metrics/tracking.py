# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import uq.stats.pdfs as uqstpdf
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
import pdb

# plt.style.use("utils/plotting/production_pyplot_style.txt")

def multiTarget_tracking_metrics(tf,groundtargetset,targetset):
    # tracking error for active times
    
        
    cols=[]
    cols.append('RMSE_Active_pos')
    cols.append('RMSE_Active_vel')
    cols.append('TIMES_Active_count')
    cols.append('TIMES_Active_percent')
    cols.append('TIMES_Active_percent_after_discovery')
    cols.append('TIMESTEPS_before_discover')
    cols.append('TIME_percent_before_discover')
    cols.append('CountGapsInactive_after_discovery')
    cols.append('MaxInactiveGap_after_discovery')
    

    
    targetnames = [groundtargetset[i].targetName for i in range(groundtargetset.ntargs)]
    
    metrics=pd.DataFrame(columns=cols,index=targetnames)
    
    DF={}
    
    for targi in range(groundtargetset.ntargs):
          
        targID = groundtargetset[targi].ID
        target = targetset.getTargetByID(targID)
        targetName = groundtargetset[targi].targetName
        targetColor = groundtargetset[targi].color
        
        tt,xtk = groundtargetset[targi].groundtruthrecorder.getvar_uptotime_list('xtk',tf,returntimes=True)
        
        if target is not None:
            tu,xuk = target.recorderpost.getvar_uptotime_list('xfk',tf,returntimes=True)
            _,Puk = target.recorderpost.getvar_uptotime_list('Pfk',tf,returntimes=True)
            tu,status = target.recorderpost.getvar_uptotime_list('status',tf,returntimes=True)
        else:
            metrics.loc[targetName,'TIMES_Active_count'] = 0
            metrics.loc[targetName,'TIMES_Active_percent'] = 0
            metrics.loc[targetName,'TIMESTEPS_before_discover'] = len(tt)
            metrics.loc[targetName,'TIME_percent_before_discover'] = 100
            metrics.loc[targetName,'TIMES_Active_percent_after_discovery'] = 0
            metrics.loc[targetName,'CountGapsInactive_after_discovery'] = np.nan
            metrics.loc[targetName,'MaxInactiveGap_after_discovery'] = np.nan
            
            continue
        
            
        # if target.isSearchTarget():
        #     continue
        
        
        
        
        
        df=pd.DataFrame(columns=['status'],index=tt)
        df['status']='InActive'
        
        for i in range(len(tu)):
            df.loc[tu[i],'status'] = status[i]
            
        
        
        dfact = df[df['status']=='Active']
        
        for t in dfact.index:
            it = tt.index(t)
            iu = tu.index(t)

            e = xtk[it]-xuk[iu]
            
            df.loc[t,'epos'] = nplinalg.norm(e[0:2])
            df.loc[t,'evel'] = nplinalg.norm(e[2:4])
            
            m = xuk[iu][0:2]
            P = Puk[iu][0:2,0:2]
            X = xtk[it][0:2]
            df.loc[t,'pos_probRatio'] = uqstpdf.gaussianPDF_propratio(X,m,P,sig=2)
            df.loc[t,'probRatio'] = uqstpdf.gaussianPDF_propratio(xtk[it],xuk[iu],Puk[iu],sig=2)
            
        # get metrics   
        e = df[df['status']=='Active']['epos'].values
        metrics.loc[targetName,'RMSE_Active_pos'] = np.sqrt(np.mean(np.power(e,2)))
        
        e = df[df['status']=='Active']['evel'].values
        metrics.loc[targetName,'RMSE_Active_vel'] = np.sqrt(np.mean(np.power(e,2)))
        
        metrics.loc[targetName,'TIMES_Active_count'] = len(df[df['status']=='Active'])
        metrics.loc[targetName,'TIMES_Active_percent'] = 100*len(df[df['status']=='Active'])/len(df)
            
        metrics.loc[targetName,'TIMESTEPS_before_discover']=df[df['status']=='Active'].index[0]
        metrics.loc[targetName,'TIME_percent_before_discover'] = 100*metrics.loc[targetName,'TIMESTEPS_before_discover']/tt[-1]
        
        indt = df[df['status']=='Active'].index[0]
        dd = df.loc[indt:,:]
        metrics.loc[targetName,'TIMES_Active_percent_after_discovery']=100*len(dd[dd['status']=='Active'])/len(dd)
        
        L=[]
        crr=dd.iloc[0]['status']
        l=[crr]
        for indd in dd['status'].index:
            if dd.loc[indd,'status']==crr:
                l.append(crr)
            else:
                L.append(l)
                crr = dd.loc[indd,'status']
                l=[crr]
                
        L.append(l)
        
        Lactive =  list( filter(lambda x: 'Active' in x,L))
        LInactive =list( filter(lambda x: 'InActive' in x,L))
        # print("------- %s ---------"%targetName)
        # print(Lactive)
        # print(LInactive)
        metrics.loc[targetName,'CountGapsInactive_after_discovery'] = len(LInactive)
        LL=[0]+[len(l) for l in LInactive]
        metrics.loc[targetName,'MaxInactiveGap_after_discovery'] = max(LL)
        
        fig1 = plt.figure("2sigma_pos_ellipsoid_ratio")
        axlist = fig1.axes
        if len(axlist)==0:
            ax1 = fig1.add_subplot(111)
            ax1.plot(tt,np.ones(len(tt)),'--',c='k')
        else:
            ax1 = axlist[0]
            # ax1.cla()
            ax1.plot(tt,np.ones(len(tt)),'--',c='k')
            
        fig2 = plt.figure("Error_in_position")
        axlist = fig2.axes
        if len(axlist)==0:
            ax2 = fig2.add_subplot(111)
        else:
            ax2 = axlist[0]
            # ax2.cla()
            
        fig3 = plt.figure("Error_in_velocity")
        axlist = fig3.axes
        if len(axlist)==0:
            ax3 = fig3.add_subplot(111)
        else:
            ax3 = axlist[0]
            # ax3.cla()
        
        
        # figure plotting errors
        ax1.plot(df.index,df['pos_probRatio'],c=targetColor)
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('p(x-true)/p(2-sigma)')
        
        

        ax2.plot(df.index,df['epos'],c=targetColor)
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('error in position (m)')
        
        
        

        ax3.plot(df.index,df['evel'],c=targetColor)
        ax3.set_xlabel('time (s)')
        ax3.set_ylabel('error in velocity (m/s)')
    
        DF[targetName] = df

    figs = [fig1,fig2,fig3]
    
    return metrics,DF,figs
    
        
        
        