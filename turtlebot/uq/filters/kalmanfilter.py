#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""


import logging
import numpy as np
import pdb
import numpy.linalg as nplg
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from uq.uqutils import pdfs as uqutlpdfs
from uq.filters._basefilter import IntegratedBaseFilterStateModel, FiltererBase

#    print("K = ",K)
#    print("zk = ",zk )
#    print("mz = ",mz)

class KFfilterer(FiltererBase):
    """
    uses 1 model, 1 gaussian mean and covariance
    """
    filterName = 'Kalman Filter filterer'

    def debugStatus(self):
        ss=[]
        ss.append(['----------','----------'])
        ss.append( ['filterName',self.filterName] )
        ss.append(['----------','----------'])
        return ss

    @staticmethod
    def propagate(t, dt,dynModel,xfk,Pfk,uk,**params):
        _, xfk1 = dynModel.propforward(t, dt, xfk, uk=uk, **params)
        Q = dynModel.processNoise(t, dt, xfk, uk=uk, **params)
        F = dynModel.F(t, dt, xfk, uk=uk, **params)

        Pfk1 = multi_dot([F, Pfk, F.T]) + Q

        return xfk1, Pfk1

    @staticmethod
    def getPDFz(t, dt,xfk,Pfk,sensormodel):
        """
        if zk is None, just do pseudo update
        return (xu, Pu, mz,R, Pxz,Pz, K, likez )
        """

        mz, isinFOV, L = sensormodel(t, dt, xfk)

        H = sensormodel.H(t, dt, xfk)
        R = sensormodel.measNoise(t, dt, xfk)

        Pz = multi_dot([H, Pfk, H.T]) + R
        pdfz = multivariate_normal(mz,Pz)
        pdfz.isInNsig= lambda x,N: uqutlpdfs.isInNsig(x,mz,Pz,N)

        return pdfz,H,R,mz,Pz

    @staticmethod
    def measUpdate(t, dt,xfk,Pfk,sensormodel,zk, **params):
        """
        if zk is None, just do pseudo update
        return (xu, Pu, mz,R, Pxz,Pz, K, likez )
        """
        R = sensormodel.measNoise(t, dt, xfk)

        if 'mz' in params and 'Pz' in params and 'Pxz' in params:
            mz = params['mz']
            Pz = params['Pz']
            Pxz = params['Pxz']
        else:
            mz, isinFOV, L = sensormodel(t, dt, xfk)

            H = sensormodel.H(t, dt, xfk)


            Pxz = np.matmul(Pfk, H.T)
            Pz = multi_dot([H, Pfk, H.T]) + R

        try:
            pdfz = multivariate_normal(mz,Pz)
            pdfz.isInNsig= lambda x,N: uqutlpdfs.isInNsig(x,mz,Pz,N)
        except:
            pdb.set_trace()
            
        if zk is None:
            xu, Pu, K = KFfilterer.KalmanDiscreteUpdate(xfk, Pfk, mz, mz, Pz, Pxz)
#            return (xu, Pu, mz,R, Pxz,Pz, K, pdfz, likez )
            return (xu, Pu, mz,R, Pxz,Pz, K, pdfz, None )
        else:
            xu, Pu, K = KFfilterer.KalmanDiscreteUpdate(xfk, Pfk, zk, mz, Pz, Pxz)

            likez = pdfz.pdf(zk)

            return (xu, Pu, mz,R, Pxz,Pz, K, pdfz, likez )


    @staticmethod
    def measWeightedBetasUpdt(t,dt,xfk,Pfk,sensormodel, betas,Zk, **params):
        """
        betas = [beta_0, beta_1, ...]
        Zk = [ zk1,zk2,...]
        beta_0 is for null measurement,i.e the target has no measurements
        """

        _, _, mz,R, Pxz,Pz, K, _,_  = KFfilterer.measUpdate(t, dt,xfk,Pfk,sensormodel,None)

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

    @staticmethod
    def KalmanDiscreteUpdate(xfk, Pfk, zk, mz, Pz, Pxz):
        K = np.matmul(Pxz, nplg.inv(Pz))
        xu = xfk + np.matmul(K, zk - mz)
        Pu = Pfk - multi_dot([K, Pz, K.T])

        return (xu, Pu,K)

    @staticmethod
    def KalmanDiscreteUpdateAngular(xfk, Pfk, zk, mz, Pz, Pxz):
        K = np.matmul(Pxz, nplg.inv(Pz))
        xu = xfk + np.matmul(K, zk - mz)
        Pu = Pfk - multi_dot([K, Pz, K.T])

        return (xu, Pu)


class EKFfilterer(KFfilterer):
    """
    uses 1 model, 1 gaussian mean and covariance
    """
    filterName = 'Extended Kalman Filter filterer'


class TargetKF:
    """
    Kalman filter for a target

    """
    filterName = 'Target Kalman Filter'
    @staticmethod
    def propagate(t, dt,target,uk, updttarget=True,justcopyprior=False,**params):
        """
        propagate from t to t+dt
        so the final state is at t+dt at the end of this method
        """

        if justcopyprior:
            xfk1, Pfk1 = target.xfk,target.Pfk 
        else:
            xfk1, Pfk1 = KFfilterer.propagate(t, dt,target.dynModel,target.xfk,target.Pfk,uk,**params)

        if updttarget:
            target.setTargetFilterStageAsPrior()
            target.updateParams(currt=t+dt,xfk=xfk1,Pfk=Pfk1)

        return (xfk1,Pfk1)

    @staticmethod
    def getPDFz(t, dt,target,sensormodel,cacheIntermediates2Target = True,**params):
        """

        """

        pdfz,H,R,mz,Pz=KFfilterer.getPDFz(t, dt,target.xfk,target.Pfk,sensormodel)

        if cacheIntermediates2Target:
            target.context['t']=t
            target.context['H']=H
            target.context['R']=R
            target.context['mz']=mz
            target.context['Pz']=Pz

        return pdfz


    @staticmethod
    def measUpdate(t, dt,target,sensormodel,zk,updttarget=True,justcopyprior=False,**params):
        """
        t is the time at which measurement update happens
        dt is just used in functions to compute something
        """
        if justcopyprior:
            xu,Pu = target.xfk,target.Pfk            
        else:
            xu, Pu, mz,R, Pxz,Pz, K, pdfz, likez  = KFfilterer.measUpdate(t, dt,target.xfk,target.Pfk,sensormodel,zk,**params)

        if updttarget:
            target.setTargetFilterStageAsPosterior()
            target.updateParams(currt=t,xfk=xu,Pfk=Pu)

        return (xu, Pu)

    @staticmethod
    def measWeightedBetasUpdt(t,dt,target,sensormodel, betas,Zk, updttarget=True, **params):
        """
        betas = [beta_0, beta_1, ...]
        Zk = [ zk1,zk2,...]
        beta_0 is for null measurement,i.e the target has no measurements
        """

        _, _, mz,R, Pxz,Pz, K,_, _  = KFfilterer.measUpdate(t, dt,target.xfk,target.Pfk,sensormodel,None,**params)

        innov1 = 0
        innov2 = 0
        for i in range(len(Zk)):
            inn = (Zk[i]-mz)
            innov1 = innov1 + betas[i+1]*inn
            innov2 = innov2 + betas[i+1]*np.outer(inn,inn)

        xu = target.xfk + np.matmul(K,innov1)
        Pc = target.Pfk-multi_dot([K, Pz, K.T])

        Pu = betas[0]*target.Pfk+(1-betas[0])*Pc + multi_dot([K,innov2 - np.outer(innov1,innov1),K.T])

        if updttarget:
            target.setTargetFilterStageAsPosterior()
            target.updateParams(currt=t,xfk=xu,Pfk=Pu)


        return (xu, Pu)


class TargetEKF(TargetKF):
    filterName = 'Target Extended Kalman Filter'



#%%    Integrated Simulator
#######################################################
class IntegratedKF(IntegratedBaseFilterStateModel):
    filterName = 'Kalman Filter'


    def __init__(self, dynModel, sensModel, recordfilterstate=False,recorderobjprior=None,
             recorderobjpost=None,currt=0,xfk=None,Pfk=None):

        super().__init__(dynModel=dynModel, sensModel=sensModel,
             recordfilterstate=recordfilterstate,recorderobjprior=recorderobjprior,
             recorderobjpost=recorderobjpost,currt=currt,xfk=xfk,Pfk=Pfk)


    def propagate(self, t, dt, uk, **params):
        super().propagate(t, dt, uk, **params)
        xfk1, Pfk1 = KFfilterer.propagate(t, dt,self.dynModel,self.xfk,self.Pfk,uk,**params)

        self.xfk = xfk1
        self.Pfk = Pfk1
        if self.recordfilterstate:
            self.recorderprior.record(t+dt,xfk=self.xfk,Pfk=self.Pfk)


        return (xfk1, Pfk1)

    def measUpdate(self,t,dt, zk, **params):
        super().measUpdate(t,dt, zk, **params)

        xu, Pu, mz,R, Pxz,Pz, K, pdfz, likez = KFfilterer.measUpdate(t, dt,self.xfk,self.Pfk,self.sensModel,zk, **params)

        self.xfk = xu
        self.Pfk = Pu
        self.zk = zk
        if self.recordfilterstate:
            self.recorderpost.record(t,xfk=self.xfk,Pfk=self.Pfk)

        return (xu, Pu)



class IntegratedEKF(IntegratedKF):
    filterName = 'Extended Kalman Filter'






