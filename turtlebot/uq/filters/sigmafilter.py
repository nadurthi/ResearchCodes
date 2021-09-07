#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Documentation for this imm module

More details.
"""


from uq.quadratures import cubatures as quadcub

from uq.filters._basefilter import IntegratedBaseFilterStateModel, FiltererBase
import numpy.linalg as nplg
import logging
import numpy as np
import pdb
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
import pdb

from uq.uqutils import pdfs as uqutlpdfs
from uq.stats import moments as uqstmom
from uq.filters import kalmanfilter as uqfkf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




class Sigmafilterer(FiltererBase):
    """
    uses 1 model, 1 gaussian mean and covariance
    """
    filterName = 'Sigma Kalman Filter filterer'

    def __init__(self, sigmamethod = quadcub.SigmaMethod.UT):
        self.sigmamethod = sigmamethod

    def debugStatus(self):
        ss=[]
        ss.append(['----------','----------'])
        ss.append( ['filterName',self.filterName+' :  '+str(self.sigmamethod) ] )
        ss.append(['----------','----------'])
        return ss

    def propagate(self, t, dt, dynModel, xfk, Pfk, uk, **params):
        
        X, W = quadcub.GaussianSigmaPtsMethodsDict[self.sigmamethod](xfk, Pfk)
        Xk1 = np.zeros(X.shape)
        for i in range(len(W)):
            _, Xk1[i, :] = dynModel.propforward(t, dt, X[i, :], uk=uk, **params)

        xfk1, Pfk1 = uqstmom.MeanCov(Xk1, W)

        Q = dynModel.processNoise(t, dt, xfk, uk=uk, **params)

        Pfk1 = Pfk1 + Q

        return xfk1, Pfk1

    def getPDFz(self, t, dt, xfk, Pfk, sensormodel):
        """
        if zk is None, just do pseudo update
        return (xu, Pu, mz,R, Pxz,Pz, K, likez )
        """

        X, W = quadcub.GaussianSigmaPtsMethodsDict[self.sigmamethod](xfk, Pfk)

        Z = np.zeros((X.shape[0], sensormodel.hn))
        for i in range(len(W)):
            Z[i, :], isinFOV, L = sensormodel(t, dt, X[i, :])

        mz, Pz = uqstmom.MeanCov(Z, W)

        R = sensormodel.measNoise(t, dt, xfk)
        Pz = Pz + R

        Pxz = np.zeros((len(xfk), sensormodel.hn))
        for i in range(len(W)):
            Pxz = Pxz + W[i] * np.outer(X[i, :] - xfk, Z[i, :] - mz)
            
        pdfz = multivariate_normal(mz, Pz)
        pdfz.isInNsig = lambda x, N: uqutlpdfs.isInNsig(x, mz, Pz, N)

        return pdfz, R, mz, Pz, Pxz

    def measUpdate(self, t, dt, xfk, Pfk, sensormodel, zk, fovSigPtFrac=0.75,**params):
        """
        If zk is None, just do pseudo update.
        
        fovSigPtFrac = 1 : domeasupdt = False (just return prior)
        fovSigPtFrac = -1 : domeasupdt = True (do update)
        
        @param: t
        """
        if 'mz' in params and 'Pz' in params and 'Pxz' in params:
            mz = params['mz']
            Pz = params['Pz']
            Pxz = params['Pxz']
        else:
            X, W = quadcub.GaussianSigmaPtsMethodsDict[self.sigmamethod](xfk, Pfk)

            Z = np.zeros((X.shape[0], sensormodel.hn))
            isinFOV=[]
            for i in range(len(W)):
                Z[i, :], isfov, L = sensormodel(t, dt, X[i, :])
                isinFOV.append(isfov)
            
            domeasupdt=True
            if sensormodel.enforceConstraint:
                if np.sum(isinFOV)>fovSigPtFrac*len(isinFOV):
                    domeasupdt=True
                else:
                    domeasupdt=False
            if domeasupdt is False:
                return (xfk, Pfk, None, None, None, None, None, None, None)
            
            mz, Pz = uqstmom.MeanCov(Z, W)

            R = sensormodel.measNoise(t, dt, xfk)
            Pz = Pz + R

            Pxz = np.zeros((len(xfk), sensormodel.hn))
            for i in range(len(W)):
                Pxz = Pxz + W[i] * np.outer(X[i, :] - xfk, Z[i, :] - mz)

        pdfz = multivariate_normal(mz, Pz)
        pdfz.isInNsig = lambda x, N: uqutlpdfs.isInNsig(x, mz, Pz, N)

        if zk is None:
            xu, Pu, K = uqfkf.KFfilterer.KalmanDiscreteUpdate(xfk, Pfk, mz, mz, Pz, Pxz)
#            return (xu, Pu, mz,R, Pxz,Pz, K, pdfz, likez )
            return (xu, Pu, mz, R, Pxz, Pz, K, pdfz, None)
        else:
            xu, Pu, K = uqfkf.KFfilterer.KalmanDiscreteUpdate(xfk, Pfk, zk, mz, Pz, Pxz)
            likez = pdfz.pdf(zk)

            return (xu, Pu, mz, R, Pxz, Pz, K, pdfz, likez)

    def measWeightedBetasUpdt(self, t, dt, xfk, Pfk, sensormodel, betas, Zk, **params):
        """
        betas = [beta_0, beta_1, ...]
        Zk = [ zk1,zk2,...]
        beta_0 is for null measurement,i.e the target has no measurements
        """
        X, W = quadcub.GaussianSigmaPtsMethodsDict[self.sigmamethod](xfk, Pfk)

        _, _, mz, R, Pxz, Pz, K, _, _ = self.measUpdate(t, dt, xfk, Pfk, sensormodel, None)

        innov1 = 0
        innov2 = 0
        for i in range(len(Zk)):
            inn = (Zk[i]-mz)
            innov1 = innov1 + betas[i+1]*inn
            innov2 = innov2 + betas[i+1]*np.outer(inn,inn)

        xu = xfk + np.matmul(K,innov1)
        Pc = Pfk-multi_dot([K, Pz, K.T])

        Pu = betas[0]*Pfk+(1-betas[0])*Pc + multi_dot([K,innov2 - np.outer(innov1,innov1),K.T])

        return (xu, Pu)

class TargetSigmaKF:
    """
    Sigma Kalman filter for a target

    """
    filterName = 'Target Sigma Kalman Filter'

    def __init__(self,sigmamethod = quadcub.SigmaMethod.UT):
        self.sigmamethod = sigmamethod
        self.sigmafilterer = Sigmafilterer(sigmamethod=sigmamethod)

    def propagate(self,t, dt,target,uk, updttarget=True,justcopyprior=False,**params):
        """
        propagate from t to t+dt
        so the final state is at t+dt at the end of this method
        """
        
        if justcopyprior:
            xfk1, Pfk1 = target.xfk,target.Pfk 
        else:
            xfk1, Pfk1 = self.sigmafilterer.propagate(t, dt,target.dynModel,target.xfk,target.Pfk,uk,**params)

        if updttarget:
            target.setTargetFilterStageAsPrior()
            target.updateParams(currt=t+dt,xfk=xfk1,Pfk=Pfk1)

        return (xfk1,Pfk1)

    
    def getPDFz(self,t, dt,target,sensormodel,cacheIntermediates2Target = True,**params):
        """

        """

        pdfz, R, mz, Pz, Pxz=self.sigmafilterer.getPDFz(t, dt,target.xfk,target.Pfk,sensormodel)

        if cacheIntermediates2Target:
            target.context['t'] = t
            target.context['Pxz'] = Pxz
            target.context['mz'] = mz
            target.context['Pz'] = Pz

        return pdfz
    
    def measUpdate(self, t, dt, target, sensormodel, zk, updttarget=True, fovSigPtFrac=0.75,justcopyprior=False,**params):
        """
        t is the time at which measurement update happens
        dt is just used in functions to compute something
        """
        # use context only time t correctly matches
        if target.context.get('t',None)==t:
            params['Pxz'] = target.context['Pxz']
            params['mz'] = target.context['mz']
            params['Pz'] = target.context['Pz']
        else:
            for kk in ['t','Pxz','mz','Pz']:
                if kk in params:
                    del params[kk]
                if kk in target.context:
                    del target.context[kk]
        if justcopyprior:
            xu,Pu = target.xfk,target.Pfk            
        else:
            xu, Pu, mz, R, Pxz, Pz, K, pdfz, likez  = self.sigmafilterer.measUpdate(t, dt,target.xfk,target.Pfk,sensormodel,zk,fovSigPtFrac=fovSigPtFrac,**params)
        
        if updttarget:
            target.setTargetFilterStageAsPosterior()
            target.updateParams(currt=t,xfk=xu,Pfk=Pu)

        return (xu, Pu)


    def measWeightedBetasUpdt(self,t,dt,target,sensormodel, betas,Zk, updttarget=True, **params):
        """
        betas = [beta_0, beta_1, ...]
        Zk = [ zk1,zk2,...]
        beta_0 is for null measurement,i.e the target has no measurements
        """

        xu, Pu  = self.sigmafilterer.measWeightedBetasUpdt(t, dt, target.xfk, target.Pfk, sensormodel, betas, Zk, **params)

        
        if updttarget:
            target.setTargetFilterStageAsPosterior()
            target.updateParams(currt=t,xfk=xu,Pfk=Pu)


        return (xu, Pu)
    

    
# %%


class IntegratedSigmaFilter(IntegratedBaseFilterStateModel):
    filterName = 'Integrated Sigma Filter'
    def __init__(self, dynModel, sensModel, recordfilterstate=False, sigmamethod=quadcub.SigmaMethod.UT):

        super().__init__(dynModel, sensModel, recordfilterstate=recordfilterstate)
        self.sigmamethod = sigmamethod
        self.sigmafilterer = Sigmafilterer(sigmamethod=sigmamethod)

    def propagate(self, t, dt, uk, **params):
        super().propagate(t, dt, uk, **params)
        xfk1, Pfk1 = self.sigmafilterer.propagate(t, dt,self.dynModel,self.xfk,self.Pfk,uk,**params)

        self.xfk = xfk1
        self.Pfk = Pfk1
        if self.recordfilterstate:
            self.recorderprior.record(t+dt,xfk=self.xfk,Pfk=self.Pfk)

        return (xfk1, Pfk1)

    def measUpdt(self,t,dt, zk, *args, **kwargs):
        super().measUpdt(zk, *args, **kwargs)

        xu, Pu, zfk,Pz,likez = self.sigmafilterer.measUpdate(t, dt,self.xfk,self.Pfk,self.sensModel,zk)

        self.xfk = xu
        self.Pfk = Pu
        self.zk = zk
        if self.recordfilterstate:
            self.recorderpost.record(t,xfk=self.xfk,Pfk=self.Pfk)

        return (xu, Pu)




