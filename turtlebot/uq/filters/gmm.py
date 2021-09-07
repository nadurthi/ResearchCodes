#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

import logging
import numpy as np
import copy

import numpy.linalg as nplg
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
from uq.quadratures import cubatures as quadcub
from physmodels import sensormodels as physm
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from uq.filters import sigmafilter as uqfsigf
from uq.filters._basefilter import IntegratedBaseFilterStateModel
from uq.gmm import gmmbase as uqgmmbase
from utils.math import geometry as utmthgeom

class GMMfilterer:
    """
    GMM
    """
    filterName = 'GMM Filter filterer'

    def __init__(self,modefilterer):
        self.modefilterer = modefilterer

    def debugStatus(self):
        ss=[]
        ss.append(['----------','----------'])
        ss.append( ['filterName',self.filterName] )
        ss.append(['----------','----------'])
        return ss


    def propagate(self,t, dt,dynModel,gmmfk,uk,inplace=True,**params):
        
        if inplace is False:
            gmmfk = gmmfk.makeCopy()
            
        for j in range(gmmfk.Ncomp):
            xfkj1, Pfkj1 = self.modefilterer.propagate(t, dt, dynModel, gmmfk.m(j), gmmfk.P(j), uk, **params)
            gmmfk.updateComp(j, m=xfkj1, P=Pfkj1)
            
        

        return gmmfk

    def getPDFz(self,t, dt,gmmfk,sensormodel,**params):
        """
        if zk is None, just do pseudo update
        return (xu, Pu, zfk,R, Pxz,Pz, K, likez )
        """

        # now meas updt at t


        mzf = []
        Pzf = []
        Pxzf = []
        for j in range(gmmfk.Ncomp):
            pdfz, R, mz, Pz, Pxz = self.modefilterer.getPDFz(t, dt,gmmfk.m(j),gmmfk.P(j),sensormodel,**params)
            mzf.append(mz)
            Pzf.append(Pz)
            Pxzf.append(Pxz)


        gmmz = uqgmmbase.GMM.fromlist(mzf,Pzf,gmmfk.getwts(),t)
        gmmz.normalizeWts()

        return gmmz,Pxzf
    
    
    def measUpdate(self,t, dt,gmmfk,sensormodel,zk,inplace=True, **params):

        if inplace is False:
            gmmuk = gmmfk.makeCopy()
        else:
            gmmuk = gmmfk
            
        Mz=[]
        PPz=[]
        Pxzf = []
        Lj=[]
        for j in range(gmmfk.Ncomp):
            if 'isSkippedComp' in params:
                if j in params['isSkippedComp']:
                    continue
                
            if 'gmmz' in params and 'Pxzf' in params:
                newparams = {}
                newparams['mz'] = params['gmmz'].m(j)
                newparams['Pz'] = params['gmmz'].P(j)
                newparams['Pxz'] = params['Pxzf'][j]
                xu, Pu, mz, R, Pxz, Pz, K, pdfz, likez = self.modefilterer.measUpdate(t, dt, gmmfk.m(j), gmmfk.P(j), sensormodel, zk, **newparams)
            else:
                xu, Pu, mz, R, Pxz, Pz, K, pdfz, likez = self.modefilterer.measUpdate(t, dt, gmmfk.m(j), gmmfk.P(j), sensormodel, zk)
            
            w = gmmfk.w(j)
            if likez is not None:
                w = likez*w
            
            gmmuk.updateComp(j, m=xu, P=Pu,w=w)
            Mz.append(mz)
            PPz.append(Pz)
            Pxzf.append(Pxz)
            Lj.append(likez)
        
        gmmuk.normalizeWts()
        
        gmmz = uqgmmbase.GMM.fromlist(Mz,PPz,gmmfk.getwts(),t)
        gmmz.normalizeWts()

        if zk is None:
            return (gmmuk, gmmz,Pxzf, None, None )
        else:
            likez = gmmz.pdf(zk)
            return (gmmuk, gmmz,Pxzf, Lj,likez)


class TargetGMM(GMMfilterer):
    """
    GMM filter for a target

    """
    filterName = 'GMM Target Filter'
    def __init__(self,modefilterer=uqfsigf.Sigmafilterer( sigmamethod=quadcub.SigmaMethod.UT) ):
        self.modefilterer = modefilterer
        self.GMMfilterer = GMMfilterer(modefilterer)
        
    def propagate(self,t, dt,target,uk, updttarget=True,**params):
        """
        propagate from t to t+dt
        so the final state is at t+dt at the end of this method
        """

        gmmfk = self.GMMfilterer.propagate(t, dt,target.dynModel,target.gmmfk,uk,**params)
        xfk,Pfk = gmmfk.meanCov()
        if updttarget:
            target.setTargetFilterStageAsPrior()
            target.updateParams(currt=t+dt,gmmfk=gmmfk,xfk=xfk,Pfk=Pfk)

        return gmmfk

    
    def getPDFz(self,t, dt,target,sensormodel,cacheIntermediates2Target = True,**params):
        """

        """

        gmmfk, gmmz,Pxzf, Lj,likez=self.GMMfilterer.measUpdate(t, dt,target.gmmfk,sensormodel,None,inplace=False, **params)

        if cacheIntermediates2Target:
            target.context['t'] = t
            target.context['gmmz'] = gmmz
            target.context['Pxzf'] = Pxzf
            

        return gmmz
    
    def measUpdate(self, t, dt, target, sensormodel, zk, updttarget=True, **params):
        """
        t is the time at which measurement update happens
        dt is just used in functions to compute something
        """
        # use context only time t correctly matches
        if target.context.get('t',None)==t:
            params['Pxzf'] = target.context['Pxzf']
            params['gmmz'] = target.context['gmmz']
        else:
            for kk in ['t','Pxzf','gmmz']:
                if kk in params:
                    del params[kk]
                if kk in target.context:
                    del target.context[kk]

        inplace = updttarget
        gmmu, gmmz,Pxzf, Lj,likez  = self.GMMfilterer.measUpdate(t, dt,target.gmmfk,sensormodel,zk,inplace=inplace,**params)

        xfk,Pfk = gmmu.meanCov()
        if updttarget:
            target.setTargetFilterStageAsPosterior()
            target.updateParams(currt=t,gmmfk=gmmu,xfk=xfk,Pfk=Pfk)

        return gmmu, gmmz,Pxzf



        
# %%






class IntegratedGMM(IntegratedBaseFilterStateModel):
    """
    Integrated GMM hahahahahah
    """
    filterName = 'GMM Filter'


    def __init__(self, dynModel, sensModel, filterer, recordfilterstate=False):

        self.filterer = filterer
        super().__init__(dynModel, sensModel, recordfilterstate=recordfilterstate)

    # @staticmethod
    # def propagateTarget(target,sensModel=None,dynmodelIdx=0,gmmIdx=0):
    #     kf= KF(target.dynModels[dynmodelIdx],sensModel,
    #             recordfilterstate=False,currtk=target.currtk,
    #             xkf=target.xfk[gmmIdx],Pkf=target.Pfk[gmmIdx])
    #     return kf

    def propagate(self, t, dt, uk, **params):
        _, xfk1 = self.dynModel.propforward(t, dt, self.xfk, uk=uk, **params)

        Q = self.dynModel.processNoise(t, dt, self.xfk, uk=uk, **params)
        F = self.dynModel.F(t, dt, self.xfk, uk=uk, **params)

        Pfk1 = multi_dot([F, self.Pfk, F.T]) + Q

        self.xfk = xfk1
        self.Pfk = Pfk1
        if self.recordfilterstate:
            self.recordHistory()

        super().propagate(t, dt, uk, **params)

        return (xfk1, Pfk1)

    def Pz(self,t,dt):
        H = self.sensModel.H(t, dt, self.xfk)
        R = self.sensModel.measNoise(t, dt, self.xfk)

        Pz = multi_dot([H, self.Pfk, H.T]) + R

        return Pz

    def Pxz(self,t,dt):

        H = self.sensModel.H(t, dt, self.xfk)
        Pxz = np.matmul(self.Pfk, H.T)
        return Pxz


    def K(self,Pz,Pxz):

        K = np.matmul(Pxz, nplg.inv(Pz))
        return K

    def measUpdt(self,t,dt, zk, *args, **kwargs):
        zfk, isinFOV, L = self.sensModel(t, dt, self.xfk)

        H = self.sensModel.H(t, dt, self.xfk)
        R = self.sensModel.measNoise(t, dt, self.xfk)

        Pxz = np.matmul(self.Pfk, H.T)
        Pz = multi_dot([H, self.Pfk, H.T]) + R

        xu, Pu = KalmanDiscreteUpdate(self.xfk, self.Pfk, zk, zfk, Pz, Pxz)

        self.xfk = xu
        self.Pfk = Pu
        self.zk = zk
        if self.recordfilterstate:
            self.recordHistory()

        super().measUpdt(zk, *args, **kwargs)

        return (xu, Pu)














