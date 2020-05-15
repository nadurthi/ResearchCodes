# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:01:07 2019

@author: Nagnanamus
"""

from numpy import linalg as LA
import numpy as np
import sys
import pdb



        
def cart2classorb(x,mu):
#    x=[x,y,vx,vy]
#    out=[a,e,i,om,Om,M]
    if x.ndim is not 1:
        sys.exit('x.ndim has to be 1')
    
    r = x[0:3]
    v = x[3:6] 
    rm = LA.norm(r)
    vm = LA.norm(v)
    a = 1/(2/rm-vm**2/mu)
    
    hbar = np.cross(r,v)
    cbar = np.cross(v,hbar)-mu*r/rm
    
    e=LA.norm(cbar)/mu
    
    hm = LA.norm(hbar)
    ih = hbar/hm;
    
    ie = cbar/(mu*e)
    
    ip = np.cross(ih,ie)
    
    i = np.arccos(ih[2])

#    pdb.set_trace()
    w = np.arctan2(ie[2],ip[2])
    
    Om = np.arctan2(ih[0],-ih[1])
    
    f = np.arccos(np.dot(ip,r)/rm)

    if np.dot(v,r)<=0:
        f=2*np.pi-f
    
    sig = np.dot(r,v)/np.sqrt(mu)
    
    E = np.arctan2(sig/np.sqrt(a),1-rm/a)
    
    M = E-e*np.sin(E)
    
    classorb = np.array([a,e,i,w,Om,M])

    return classorb



def cart2classorb_2D(x,mu):
    y = np.array([ x[0],x[1],0,x[2],x[3], 0 ])
    classorb = cart2classorb(y,mu)

    return classorb




    
     