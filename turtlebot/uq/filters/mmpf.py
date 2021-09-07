#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

import numpy.linalg as nplg
import logging
import numpy as np
from numpy.linalg import multi_dot
import copy
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from uq.filters._basefilter import IntegratedBaseFilterStateModel, FiltererBase
from uq.filters import kalmanfilter as uqkf
from uq.gmm import gmmbase as uqgmmbase
from uq.uqutils import recorder as uqrecorder
from uq.uqutils import helper as uqutilhelp
from uq.filters import pf as uqpf
# import clc

class MMParticleKF(uqpf.Particles):
    """
    Each particle has the model state r and the corresponding Kalman Filter
    Each particle has the corresponding weight.
    X is the particle that has the states
    """
    def getEst(self):
        m=0
        P=0
        for i in self.itersamplesIdx():
            m = m + self.wts[i]*self.X[i]['xfk']
        for i in self.itersamplesIdx():
            P = P + self.wts[i]*( self.X[i]['Pfk'] + np.outer(self.X[i]['xfk']-m,self.X[i]['xfk']-m) )
        
        return m,P
    
class MMPF(FiltererBase):
    """
    uses 1 model, 1 gaussian mean and covariance
    """
    filterName = 'MM Particle Filter filterer'

    def __init__(self,modefilterer):
        self.modefilterer = modefilterer

    def debugStatus(self):
        ss=[]
        ss.append(['----------','----------'])
        ss.append( ['filterName',self.filterName] )
        ss.append(['----------','----------'])
        return ss


    def propagate(self,t, dt,dynMultiModels,particles,uk,inplace=True,**params):
        # propaagate from t to t+dt
        # SampleWts = [ [Particle,wt], [Particle,wt], [Particle,wt], [Particle,wt] ....   ]
        # p[i,j] is to go from ith state to jth state
        # X['r'] in the particle is an integer for the index of the model
        
        nm = dynMultiModels.Nmodels
        if inplace is False:
            particles_tdt = particles.makeCopy()
        else:
            particles_tdt = particles
            
        # first sample from p(rt+1|rt)
        for i in particles_tdt.itersamplesIdx():
            X, w = particles_tdt[i]
            ridx = X['r']
            pmf = dynMultiModels.p[ridx,:]
            rtdt = np.random.choice(nm,size=1, replace=True, p=pmf)
            rtdt = rtdt[0]
            
            # Now apply the modefilterer to 
            m,P = self.modefilterer.propagate(t, dt,dynMultiModels[rtdt],
                                    X['xfk'],X['Pfk'],uk,**params)
            
            particles_tdt.X[i]['r'] = rtdt
            particles_tdt.X[i]['xfk'] = m
            particles_tdt.X[i]['Pfk'] = P
            particles_tdt.wts[i] = w
            
        
        

        return particles


    def getPDFz(self,t, dt,gmmfk,modelprobfk,sensormodel,**params):
        """
        if zk is None, just do pseudo update
        return (xu, Pu, zfk,R, Pxz,Pz, K, likez )
        """
        raise NotImplementedError("Need to work on this")
        
        # now meas updt at t


        mzf = []
        Pzf = []
        for j in range(gmmfk.Ncomp):
            _,_,_,mz,Pz = self.modefilterer.getPDFz(t, dt,gmmfk.m(j),gmmfk.P(j),sensormodel,**params)
            mzf.append(mz)
            Pzf.append(Pz)



        gmmz = uqgmmbase.GMM.fromlist(mzf,Pzf,modelprobfk,t)
        gmmz.normalizeWts()

        return gmmz

    def measUpdate(self,t, dt,particles,sensormodel,zk,inplace=True,**params):
        """
        if zk is None, just do pseudo update
        meas update at t+Dt
        """

        if inplace is False:
            particles_tdt = particles.makeCopy()
        else:
            particles_tdt = particles

        # now meas updt at t
        for i in particles_tdt.itersamplesIdx():
            X, w = particles_tdt[i]
            
            m, P, mz,R, Pxz,Pz, K, pdfz, likez = self.modefilterer.measUpdate(t, dt,X['xfk'],X['Pfk'],sensormodel,zk)
            
            
            particles_tdt.X[i]['xfk'] = m
            particles_tdt.X[i]['Pfk'] = P
            particles_tdt.wts[i] = likez*w
            
        
        particles_tdt.renormlizeWts()

        ww=np.array(particles_tdt.wts)
        Neff = 1/( np.sum(ww**2) )
        if Neff<0.5*particles_tdt.Nsamples:
            print("bootstrap filtering")
        particles_tdt.bootstrapResample()

        return particles_tdt
    




class TargetMMPF:
    """
    IMM filter for a target

    """
    filterName = 'Target MM-PF Filter'

    def __init__(self,modefilterer):
        self.modefilterer = modefilterer
        self.mmpffilterer=MMPF(self.modefilterer)

    def propagate(self,t, dt,target,uk, updttarget=True,**params):
        """
        propagate from t to t+dt
        so the final state is at t+dt at the end of this method
        """


        particles = self.mmpffilterer.propagate(t, dt,target.dynModelset,target.particlesfk,uk,inplace=True,**params)

        if updttarget:
            target.setTargetFilterStageAsPrior()
            xfk,Pfk = particles.getEst()
            target.updateParams(currt=t+dt,particlesfk=particles,xfk=xfk,Pfk=Pfk)

        return particles


    def getPDFz(self,t, dt,target,sensormodel,cacheIntermediates2Target = True,**params):
        """

        """
        raise NotImplementedError("Need to work on this")
        
#        pdfz =self.mmpffilterer.getPDFz(t, dt,target.gmmfk,target.modelprobfk,sensormodel)

        gmmu, modelprobu, gmmz,Pxzf, _ = self.mmpffilterer.measUpdate(t, dt,target.gmmfk,target.modelprobfk,sensormodel,None,inplace=False,**params)

        if cacheIntermediates2Target:
            target.context['t']=t
            target.context['gmmz']=gmmz
            target.context['Pxzf']=Pxzf


        return gmmz



    def measUpdate(self,t, dt,target,sensormodel,zk,updttarget=True,**params):
        """
        t is the time at which measurement update happens
        dt is just used in functions to compute something
        """

        # use context only time t correctly matches
        # if target.context.get('t',None)==t:
        #     params['Pxzf'] = target.context['Pxzf']
        #     params['gmmz'] = target.context['gmmz']
        # else:
        #     for kk in ['t','Pxzf','gmmz']:
        #         if kk in params:
        #             del params[kk]
        #         if kk in target.context:
        #             del target.context[kk]

        inplace = updttarget
        particlesfu  = self.mmpffilterer.measUpdate(t, dt,target.particlesfk,sensormodel,zk,inplace=inplace,**params)

        if updttarget:
            target.setTargetFilterStageAsPosterior()
            xfu,Pfu = particlesfu.getEst()
            target.updateParams(currt=t,particlesfk=particlesfu,xfk=xfu,Pfk=Pfu)

        return particlesfu




#%%  ---------------------------------------
class IntegratedIMM(IntegratedBaseFilterStateModel):
    """
    IMM filter for a single target, testing.
    """
    filterName = 'IMM Integrated Filter for single target'

    def __init__(self, dynMultiModels,currt,gmmfk,modelprobfk, sensormodel, modefilterer,
                 recorderobjprior=None, recorderobjpost=None,
                 recordfilterstate=False):
        super().__init__(dynModel=None, sensModel=sensormodel,
             recordfilterstate=recordfilterstate,recorderobjprior=None,
             recorderobjpost=None,currt=currt,xfk=None,Pfk=None)

        self.modefilterer = modefilterer
        self.mmpffilterer=MMPF(self.modefilterer)

        self.dynMultiModels = dynMultiModels
        self.sensormodel = sensormodel


        self.recordfilterstate = recordfilterstate

        if recorderobjprior is None:
            self.recorderprior = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','gmmfk','modelprobfk'] )
        else:
            self.recorderprior = recorderobjprior

        if recorderobjpost is None:
            self.recorderpost = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk','gmmfk','modelprobfk'] )
        else:
            self.recorderpost = recorderobjpost

        # gmmfk weights are always equal
        self.gmmfk = gmmfk
        self.modelprobfk = modelprobfk
        self.currt = currt
        xfk,Pfk = self.mmpffilterer.getEst(gmmfk,modelprobfk)

        self.groundtruthrecorder = uqrecorder.StatesRecorder_list(statetypes = ['xfk'] )

        if self.recordfilterstate:
            self.recorderprior.record(currt,gmmfk=self.gmmfk, modelprobfk=self.modelprobfk,xfk=xfk,Pfk=Pfk)
            self.recorderpost.record(currt,gmmfk=self.gmmfk, modelprobfk=self.modelprobfk,xfk=xfk,Pfk=Pfk)

    def setInitialSingledata(self,gmmfk,modelprobfk,currt):
        self.gmmfk = gmmfk
        self.currt = currt
        self.modelprobfk = modelprobfk
        xfk,Pfk = self.mmpffilterer.getEst(gmmfk,modelprobfk)
        if self.recordfilterstate:
            self.recorderprior.record(currt,gmmfk=self.gmmfk, modelprobfk=self.modelprobfk,xfk=xfk,Pfk=Pfk)
            self.recorderpost.record(currt,gmmfk=self.gmmfk, modelprobfk=self.modelprobfk,xfk=xfk,Pfk=Pfk)



    def propagate(self,t,dt,uk, **params):
        super().propagate(t, dt, uk, **params)


        gmmfk, modelprobfk = self.mmpffilterer.propagate(t, dt,self.dynMultiModels,self.gmmfk,self.modelprobfk,uk,**params)
        self.gmmfk = gmmfk
        self.modelprobfk = modelprobfk

        if self.recordfilterstate:
            xfk,Pfk = self.mmpffilterer.getEst(self.gmmfk,self.modelprobfk)
            self.recorderprior.record(t+dt,modelprobfk=self.modelprobfk,gmmfk=self.gmmfk,xfk = xfk,Pfk = Pfk)


    def measUpdate(self,t,dt, zk, **params):
        super().measUpdate(t,dt, zk, **params)

        gmmu, modelprobu, gmmz,Pxzf, Lj  = self.mmpffilterer.measUpdate(t, dt,self.gmmfk,self.modelprobfk,self.sensormodel,zk,inplace=True,**params)
        self.gmmfk = gmmu
        self.modelprobfk = modelprobu

        if self.recordfilterstate:
            xfk,Pfk = self.mmpffilterer.getEst(self.gmmfk,self.modelprobfk)
            self.recorderpost.record(t,modelprobfk=self.modelprobfk,gmmfk=self.gmmfk,xfk = xfk,Pfk = Pfk)



    def debugStatus(self):
        ss=[]
        ss.append( ['filterName',self.filterName] )
        ss.append( ['currtk',self.currtk] )
        ss.append( ['recordfilterstate',self.recordfilterstate] )
        ss.append( ['filterstage',self.filterstage] )
        ss = ss + self.dynModel.debugStatus()
        ss = ss + self.sensormodel.debugStatus()

        return ss

