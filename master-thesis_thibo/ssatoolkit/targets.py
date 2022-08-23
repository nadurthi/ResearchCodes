# -*- coding: utf-8 -*-

import scipy
import math
import numpy as np
from numpy import linalg as nplinalg
from scipy import linalg as sclinalg
import pandas as pd
import ssatoolkit.constants as cst 
#import laplace_method as iod
#import differentiate as od
# from ssatoolkit import propagators
# from prop_odeint import propagate
import uuid
from ssatoolkit import coords
import math
import numpy as np
import pickle as pkl
import sys
import collections as clc

from ssatoolkit import date_time,propagators



Meas = clc.namedtuple("Meas", "sensID zk k targID nodeID measID")


def get_OE_from_TLE(filename,mu):
    """ Extract and compute the orbital elements from raw TLE data.

    Parameters
    ----------

    data: dataframe
      A dataframe that contains a bunch of TLE from celestrack.

    Returns
    -------

    data_OE: dataframe
      a set of orbital elements.

    """
    data = pd.read_csv(filename)
    data_OE = pd.DataFrame()
    for index, row in data.iterrows():
        [E, f]=propagators.kepler(row["MEAN_ANOMALY"] * math.pi / 180, row["ECCENTRICITY"], 1e-12)
        # E = propagators.Find_Kep_E(row["ECCENTRICITY"], row["MEAN_ANOMALY"] * math.pi / 180)
        # f = np.arccos((np.cos(E) - row["ECCENTRICITY"]) / (1 - row["ECCENTRICITY"] * np.cos(E)))

        # if (int(E / math.pi) % 2) == 1:
        #     f = 2 * math.pi - f
        n = row["MEAN_MOTION"]
        # a= (mu/n**2)**(1/3)
        a = (mu / (2 * math.pi * row["MEAN_MOTION"] / 86400) ** 2) ** (1 / 3)
        e = row["ECCENTRICITY"]
        i = row["INCLINATION"] * math.pi / 180
        Om = row["RA_OF_ASC_NODE"] * math.pi / 180
        om = row["ARG_OF_PERICENTER"] * math.pi / 180
        # f = f * math.pi / 180
        orbital_elements = pd.DataFrame({'a':[a], 'e':[e], 'i':[i], 'Om':[Om], 'om':[om], 'f':[f], 'Epoch':[row["EPOCH"]]})
        data_OE = pd.concat([data_OE,orbital_elements], ignore_index=True, axis=0)
    
    data_OE.reset_index(inplace=True)
    return data_OE




class TrueTargets:
    def __init__(self, mu):
        
        self.mu = mu 
        self.Nmax=None
    def get_pickle(self,filename):
        with open(filename,'rb') as F:
            data = pkl.load(F)
        
        self.true_catalogue = data['df']
        self.Nmax=self.true_catalogue.shape[0]
        self.targIds = self.true_catalogue.index
        
        return data['t0'],data['dt'],data['tf'],data['Tvec']
        
        
        
        
    def getCSV(self,filename):
        self.filename = filename
        self.true_catalogue = get_OE_from_TLE(self.filename,self.mu)
        self.Nmax=self.true_catalogue.shape[0]
        self.targIds = self.true_catalogue.index
        
    def getCSVcustom(self):
        self.true_catalogue = pd.read_csv(self.filename)
        self.Nmax=self.true_catalogue.shape[0]
        self.targIds = self.true_catalogue.index
        
    def generate_trajectory(self, Tvec):
        self.true_trajectory = {}
        for idx in self.targIds:
            Xcart = np.zeros((len(self.true_catalogue), 6))
            orb_elem = self.true_catalogue.loc[idx, ['a','e','i','Om','om','f']]
            rv0 = coords.from_OE_to_RV(orb_elem, self.mu)
            Xcart = propagators.propagate_FnG(Tvec, rv0,self.mu)
            self.true_trajectory[idx] = Xcart
    
    def setMaxTargets(self,Nmax):
        self.Nmax = Nmax
        self.targIds = self.true_catalogue.index[:Nmax]
        
    def get_meas_sens_target(self,targidx,sens,tks,Tvec,withNoise=True):
        Zk=[]
        Tk=[]
        for k in tks:
            xk = self.true_trajectory[targidx][k, :]
            if withNoise is True:
                zk = sens.gen_meas(xk)
            else:
                zk = sens.evalfunc(xk)
                
            if zk is not None:
                Zk.append(zk)
                Tk.append(Tvec[k])
        return Tk,Zk
    
    def gettargetIdxs(self):
        return list( self.targIds )
    
    def getOrbs(self):
        Orbs=[]
        for idx in self.targIds:
            orb = self.true_catalogue.loc[idx, ['a','e','i','Om','om','f']].values
            Orbs.append(orb)
        return np.vstack(Orbs)
        
    def get_measurements(self, sensors,k):
        M = []
        for idx in self.targIds:
            xk = self.true_trajectory[idx][k, :]
            for sens in sensors.itersensors():
                zk = sens.gen_meas(xk)
                if zk is not None:
                    M.append(Meas(sens.idx, zk, k,idx,None,None))

        return M
    
    def getMeasStats(self,sensors,t0, tf, dt,NminMeas = 10):
        Tvec = np.arange(t0, tf, dt)
        MeasStats={}
        for idx in self.targIds:
            MeasStats[idx] = np.zeros(len(Tvec))
            for k in range(len(Tvec)):
                xk = self.true_trajectory[idx][k, :]
                for sens in sensors.itersensors():
                    if sens.inFOV(xk):
                        MeasStats[idx][k]+=1
    
        self.MeasStats=MeasStats
        # drop trajectories that have fewer than 10 measurments in total over the whole Tvec simulation duration
        targIds = list(self.true_trajectory.keys())
        for idx in targIds:
            if np.sum(MeasStats[idx])<NminMeas:
                self.true_trajectory.pop(idx)
                self.true_catalogue.drop(idx,inplace=True)
        
        self.targIds=self.true_catalogue.index[:self.Nmax]
        
class CatalogTargets:
    def __init__(self):
        self.catalog=[]
        
    def addTarget_orbElem(self,orbElem,targID=uuid.uuid4()):
        target={'id':targID,'orbElem':orbElem,}
        
        self.catalog.append(target)
        

class GenOrbits:
    def __init__(self,R,mu):
        self.R=R
        self.mu=mu
        self.orbElems=[]
        # 100,sensors,dt,10,[7500,12000],[0.0001,0.9],[0.1,0.9*np.pi],[0.1,0.9*np.pi],[0.1,0.9*np.pi],[0,2*np.pi]
    def genOrbits(self,Norbs,sensors,Tvec,minMeas,alimits,elimits,ilimits,Omlimits,omlimits,flimits):
        cnt=0
        while 1:
            
            a=alimits[0]+(alimits[1]-alimits[0])*np.random.rand(1)[0]
            e=elimits[0]+(elimits[1]-elimits[0])*np.random.rand(1)[0]
            i=ilimits[0]+(ilimits[1]-ilimits[0])*np.random.rand(1)[0]
            Om=Omlimits[0]+(Omlimits[1]-Omlimits[0])*np.random.rand(1)[0]
            om=omlimits[0]+(omlimits[1]-omlimits[0])*np.random.rand(1)[0]
            f=flimits[0]+(flimits[1]-flimits[0])*np.random.rand(1)[0]
            # Tp = 2*np.pi*np.sqrt(a**3/mu)
            orb = [a,e,i,Om,om,f]
            
            if a*(1-e)<self.R:
                continue
            
            rv = coords.from_OE_to_RV(orb,self.mu)
            T=2*np.pi*np.sqrt(a**3/self.mu)
            if T>Tvec[int(len(Tvec)/2)]:
                continue
            
            X=propagators.propagate_orb(Tvec[Tvec<=(T+10*60)], orb,self.mu)
             
            flg=False
            Nz=np.zeros(len(sensors))
            for scnt,sens in enumerate(sensors.itersensors()):
                for j in range(X.shape[0]): 
                    if sens.inFOV(X[j]):
                        Nz[scnt]+=1
                    
                    if any(Nz>minMeas):
                        flg=True
                        break
                
                if flg:
                    break
            
            
            if flg:
                self.orbElems.append([a,e,i,Om,om,f])
            else:
                continue
            
            if cnt>=Norbs:
                break
            cnt+=1
            print(cnt)
        self.orbElems_df = pd.DataFrame(data=self.orbElems,columns=['a','e','i','Om','om','f'])
        return self.orbElems_df
    
    def save(self,filepath):
        self.orbElems_df.to_csv(filepath)
            
            
            
            
            
            
            
            
            
            
            
            
            
    