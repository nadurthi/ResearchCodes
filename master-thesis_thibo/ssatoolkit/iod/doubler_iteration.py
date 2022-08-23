# -*- coding: utf-8 -*-
import math
import numpy as np
# import toolkit as tl
# import radar_filter
from ssatoolkit import propagators
# import pandas as pd
# import os
import pdb
from numpy import linalg as nplinalg
import warnings

def func(t0,t2,rmag0,rmag1,c0,c1,r_site,L,tm,mu):
    rsite_mag = nplinalg.norm(r_site) 
    rho0 = 0.5*(-c0+np.sqrt( c0**2-4*(rsite_mag**2-rmag0**2) ))
    rho1 = 0.5*(-c1+np.sqrt( c1**2-4*(rsite_mag**2-rmag1**2) ))
    
    rbar0 = rho0*L[0]+r_site
    rbar1 = rho1*L[1]+r_site
    
    What = np.cross(rbar0,rbar1)/(nplinalg.norm(rbar0)*nplinalg.norm(rbar1) )
    
    rho2 = -np.dot(r_site,What)/(np.dot(L[2],What))
    
    rbar2 = rho2*L[2]+r_site
    
    rbars = [rbar0,rbar1,rbar2]
    
    COS_delv={}
    SIN_delv={}
    aTAN_delv={}
    for j in [1,2]:
        for k in [0,1]:
            COS_delv[(j,k)] =np.dot(rbars[j],rbars[k])/(nplinalg.norm(rbars[j])*nplinalg.norm(rbars[k])) 
            SIN_delv[(j,k)] = tm*np.sqrt(1-COS_delv[(j,k)]**2)
            
            aTAN_delv[(j,k)] = np.arctan2(SIN_delv[(j,k)] ,COS_delv[(j,k)])
            if aTAN_delv[(j,k)]<0:
                aTAN_delv[(j,k)] = aTAN_delv[(j,k)]+2*np.pi
            
    rmag0 = nplinalg.norm(rbar0)
    rmag1 = nplinalg.norm(rbar1)
    rmag2 = nplinalg.norm(rbar2)
    
    if aTAN_delv[(2,0)] > np.pi:
        c0 = rmag1 * SIN_delv[(2,1)] / (rmag0*SIN_delv[(2,0)]  ) 
        c2 = rmag1 * SIN_delv[(1,0)] / (rmag2*SIN_delv[(2,0)]  ) 
        p  = (c0*rmag0+c2*rmag2-rmag1)/(c0+c2-1)
    else:
        c0 = rmag0 * SIN_delv[(2,0)] / (rmag1*SIN_delv[(2,1)]  ) 
        c2 = rmag0 * SIN_delv[(1,0)] / (rmag2*SIN_delv[(2,1)]  ) 
        p  = (c2*rmag2-c0*rmag1+rmag0)/(-c0+c2+1)
    
    eCOSv0 = p/rmag0-1
    eCOSv1 = p/rmag1-1
    eCOSv2 = p/rmag2-1
    
    
    if aTAN_delv[(1,0)] !=np.pi:
        eSINv1 = (-COS_delv[(1,0)]*eCOSv1 + eCOSv0)/SIN_delv[(1,0)]
    else:
        eSINv1 = (COS_delv[(2,1)]*eCOSv1 - eCOSv2)/SIN_delv[(2,0)]
        
    e=np.sqrt(eCOSv1**2+eSINv1**2 )
    
    a=p/(1-e**2)
    n=np.sqrt(mu/a**3)
    
    S=(rmag1/p)*np.sqrt(1-e**2)*eSINv1
    C=(rmag1/p)*(e**2+eCOSv1)
    
    SIN_DEL_E21 = ( rmag2/np.sqrt(a*p) )*SIN_delv[(2,1)] - ( rmag2/p )*(1-COS_delv[(2,1)])*S
    
    COS_DEL_E21 = 1 - (rmag1*rmag2/(a*p))*(1-COS_delv[(2,1)])
    
    aTAN_DEL_E21 = np.arctan2(SIN_DEL_E21 ,COS_DEL_E21)
    if aTAN_DEL_E21<0:
        aTAN_DEL_E21 = aTAN_DEL_E21+2*np.pi
    
    SIN_DEL_E10 =  ( rmag0/np.sqrt(a*p) )*SIN_delv[(1,0)] + ( rmag0/p )*(1-COS_delv[(1,0)])*S
    
    COS_DEL_E10 = 1 - (rmag1*rmag0/(a*p))*(1-COS_delv[(1,0)])
    
    aTAN_DEL_E10 = np.arctan2(SIN_DEL_E10 ,COS_DEL_E10)
    if aTAN_DEL_E10<0:
        aTAN_DEL_E10 = aTAN_DEL_E10+2*np.pi
    
    
    DEL_M21 = aTAN_DEL_E21+2*S*(np.sin(aTAN_DEL_E21/2))**2-C*SIN_DEL_E21
    DEL_M01 = -aTAN_DEL_E10+2*S*(np.sin(aTAN_DEL_E10/2))**2+C*SIN_DEL_E10
    
    F0 = t0-DEL_M01/n
    F1 = t2-DEL_M21/n
    
    params={'SIN_DEL_E21':SIN_DEL_E21,'COS_DEL_E21':COS_DEL_E21,
            'aTAN_DEL_E21':aTAN_DEL_E21,'a':a,'rbars':rbars}
    
    return F0,F1,params


def double_r_iteration(r_site, T, L, Tchecks,Lchecks,mu,tol):
    warnings.filterwarnings("ignore")

    tm=1
    
    t0=T[0]-T[1]
    t2=T[2]-T[1]
    
    rmag0 = 12756.274
    rmag1 = 12820.055
    
    c0 = 2*np.dot(L[0],r_site)
    c1 = 2*np.dot(L[1],r_site)
    # rsite_mag = nplinalg.norm(r_site)
    Qprev=1000000
    while True:
        F0,F1,params = func(t0,t2,rmag0,rmag1,c0,c1,r_site,L,tm,mu)
        
        SIN_DEL_E21 = params['SIN_DEL_E21']
        COS_DEL_E21 = params['COS_DEL_E21']
        aTAN_DEL_E21 = params['aTAN_DEL_E21']
        a = params['a']
        rbars = params['rbars']
        [rbar0,rbar1,rbar2]=rbars
        
        Q=np.sqrt(F0**2+F1**2)
        
        if np.abs(Q-Qprev)<tol:
            f = 1-(a/rmag1)*(1-COS_DEL_E21)
            g = t2-np.sqrt(a**3/mu)*(aTAN_DEL_E21-SIN_DEL_E21)
            
            vbar1 = (rbar2-f*rbar1)/g
            
            return np.hstack([rbar1,vbar1])
        if np.isnan(Q):
            return None
        
        Qprev=Q
        
        delrmag0 = 0.005*rmag0
        delrmag1 = 0.005*rmag1
        
        F0delr0,F1delr0,_ = func(t0,t2,rmag0+delrmag0,rmag1,c0,c1,r_site,L,tm,mu)
        delF0_delrmag0 = (F0delr0-F0)/delrmag0
        delF1_delrmag0 = (F1delr0-F1)/delrmag0
        
        F0delr1,F1delr1,_ = func(t0,t2,rmag0,rmag1+delrmag1,c0,c1,r_site,L,tm,mu)
        delF0_delrmag1 = (F0delr1-F0)/delrmag1
        delF1_delrmag1 = (F1delr1-F1)/delrmag1
        
        DEL = delF0_delrmag0*delF1_delrmag1-delF1_delrmag0*delF0_delrmag1        
        
        DEL0 = delF1_delrmag1*F0-delF0_delrmag1*F1
        DEL1 = delF0_delrmag0*F1-delF1_delrmag0*F0
        
        delrmag0=-DEL0/DEL
        delrmag1=-DEL1/DEL
        
        rmag0 = rmag0+delrmag0
        rmag1 = rmag1+delrmag1
        
        
        
        # print(Q,rmag0,rmag1)
        
        
        
            
            
            
            
            
            
            
            
            
            
            
