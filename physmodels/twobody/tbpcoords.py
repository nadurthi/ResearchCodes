# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:01:07 2019

@author: Nagnanamus
"""

from numpy import linalg as LA
import numpy as np
import sys
import pdb
import physmodels.twobody.constants as tbpconst



def convertCartState2Dto3D(X):
    pass

def convertCartState3Dto2D(X):
    pass


class DimensionalConverter:
    def __init__(self,planetconst):
        self.planetconst = planetconst
        
        self.mu = self.planetconst.getTrue().mu
        self.RU     = self.planetconst.getTrue().R
        self.TU     = np.sqrt(self.RU**3 / self.mu);
        self.VU     = self.RU/self.TU
        
        self.trueA2normA=self.TU**2/self.RU
        self.normA2trueA=self.RU/self.TU**2;
        
        self.trueV2normV=self.TU/self.RU
        self.normV2trueV=self.RU/self.TU
        
        self.trueX2normX=1/self.RU
        self.normX2trueX=self.RU
        
        self.trueT2normT=1/self.TU
        self.normT2trueT=self.TU

    def true2can_acc(self,X):
        return self.trueA2normA*X
    
    def true2can_pos(self,X):
        return self.trueX2normX*X
    
    def true2can_vel(self,X):
        return self.trueV2normV*X
    def true2can_posvel(self,X):
        if X.ndim==1:
            n=X.size()/2
            return np.hstack( [self.trueX2normX*X[0:n],self.trueV2normV*X[0:n]])
        else:
            n=X.shape[1]/2
            return np.hstack( [self.trueX2normX*X[:,0:n],self.trueV2normV*X[:,0:n]])
        
    def true2can_time(self,X):
        return self.trueT2normT*X


class CoordConverter:
    """Helper for coordinate transformations."""

    def __init__(self,planetconst):
        self.planetconst = planetconst
        
    def cart2orb(X):
        pass
    def orb2cart(X):
        pass
    
    

# %%  Main Coordinate Transformation functions
   
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




    
     