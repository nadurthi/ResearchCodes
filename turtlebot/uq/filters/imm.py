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




class IMMfilterer(FiltererBase):
    """
    uses 1 model, 1 gaussian mean and covariance
    """
    filterName = 'IMM Filter filterer'

    def __init__(self,modefilterer):
        self.modefilterer = modefilterer

    def debugStatus(self):
        ss=[]
        ss.append(['----------','----------'])
        ss.append( ['filterName',self.filterName] )
        ss.append(['----------','----------'])
        return ss


    def propagate(self,t, dt,dynMultiModels,gmmfk,modelprobfk,uk,inplace=True,**params):
        # first mix at time t and then propagate means/cov separately
        # to time t+dt

        if inplace is False:
            gmmfk = gmmfk.makeCopy()

        nm = dynMultiModels.Nmodels
        muij = np.zeros((nm,nm)) # (from,to)  (i,j)  (k-1,k-1)
        for i in range(nm):
            for j in range(nm):
                muij[i,j]= modelprobfk[i]*dynMultiModels.p[i,j]

        # update model probs to time t+dt
        modelprobfk1 = np.sum(muij,axis=0)
        muij = np.divide(muij,modelprobfk1)

        # mixing at time t

        for j in range(gmmfk.Ncomp):
            mj,Pj = gmmfk.weightedest(muij[:,j])
            gmmfk.updateComp(j,m=mj,P=Pj)

        # now propagate using the model dynamics to t+dt
        for j in range(gmmfk.Ncomp):
            m,P = self.modefilterer.propagate(t, dt,dynMultiModels[j],
                                    gmmfk.m(j),gmmfk.P(j),
                                    uk,**params)
            gmmfk.updateComp(j,m=m,P=P)

        gmmfk.normalizeWts()
        modelprobfk1 = modelprobfk1/np.sum(modelprobfk1)

        return gmmfk, modelprobfk1


    def getPDFz(self,t, dt,gmmfk,modelprobfk,sensormodel,**params):
        """
        if zk is None, just do pseudo update
        return (xu, Pu, zfk,R, Pxz,Pz, K, likez )
        """

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

    def measUpdate(self,t, dt,gmmfk,modelprobfk,sensormodel,zk,inplace=True,**params):
        """
        if zk is None, just do pseudo update
        """

        if inplace is False:
            gmmu = gmmfk.makeCopy()
            modelprobu = modelprobfk.copy()
        else:
            gmmu = gmmfk
            modelprobu = modelprobfk

        # now meas updt at t
        Lj=[]
        mzf = []
        Pzf = []
        Pxzf = []
        for j in range(gmmfk.Ncomp):
            if 'gmmz' in params and 'Pxzf' in params:
                newparams = {}
                newparams['mz'] = params['gmmz'].m(j)
                newparams['Pz'] = params['gmmz'].P(j)
                newparams['Pxz'] = params['Pxzf'][j]
                m, P, mz,R, Pxz,Pz, K, pdfz, likez = self.modefilterer.measUpdate(t, dt,gmmfk.m(j),gmmfk.P(j),sensormodel,zk,**newparams)
            else:
                m, P, mz,R, Pxz,Pz, K, pdfz, likez = self.modefilterer.measUpdate(t, dt,gmmfk.m(j),gmmfk.P(j),sensormodel,zk)
            gmmu.updateComp(j,m=m,P=P)

            mzf.append(mz)
            Pzf.append(Pz)
            Pxzf.append(Pxz)
            if likez is not None:
                Lj.append( likez  )
                modelprobu[j] = likez*modelprobfk[j]

        gmmz = uqgmmbase.GMM.fromlist(mzf,Pzf,modelprobfk,t)
        gmmz.normalizeWts()
        gmmu.normalizeWts()

        # update model probs at t
        modelprobu = modelprobu/np.sum(modelprobu)

        if zk is None:
            return (gmmu, modelprobu, gmmz,Pxzf, None )
        else:
            likez = gmmz.pdf(zk)
            return (gmmu, modelprobu, gmmz,Pxzf, Lj )


    def measWeightedBetasUpdt(self,t,dt,gmmfk,modelprobfk,sensormodel, betas,Zk,inplace=True, **params):
        """
        betas = [beta_0, beta_1, ...]
        Zk = [ zk1,zk2,...]
        beta_0 is for null measurement,i.e the target has no measurements
        """

        if inplace is False:
            gmmu = gmmfk.makeCopy()
            modelprobu = modelprobfk.copy()
        else:
            gmmu = gmmfk
            modelprobu = modelprobfk



        modelprobu = modelprobfk*betas[0]

        alphas = np.zeros((gmmfk.Ncomp,len(betas) ))
        alphas[:,0] = modelprobfk*betas[0]


        if 'gmmz' in params and 'Pxzf' in params:
            gmmz = params['gmmz']
            Pxzf = params['Pxzf']
        else:
            _, _, gmmz,Pxzf, _  = self.measUpdate(t, dt,gmmfk,modelprobfk,sensormodel,None,inplace=False,**params)

        for i in range(len(Zk)):
            likeZ = gmmz.evalcomp(Zk[i],gmmz.idxs)
            likeZ=likeZ+1e-10

            modelprobui = np.multiply(likeZ, modelprobfk)
            modelprobui = modelprobui/np.sum(modelprobui)

            modelprobu = modelprobu+betas[i+1]*modelprobui

            alphas[:,i+1] = betas[i+1]*modelprobui

        alphas = np.divide(alphas,modelprobu[:,np.newaxis])




        for j in range(gmmfk.Ncomp):

            innov1 = 0
            innov2 = 0
            zfk = gmmz.m(j)
            for i in range(len(Zk)):
                inn = (Zk[i]-zfk)
                innov1 = innov1 + alphas[j,i+1]*inn
                innov2 = innov2 + alphas[j,i+1]*np.outer(inn,inn)

            Pz = gmmz.P(j)
            K = np.matmul( Pxzf[j], nplg.inv(Pz) )


            xu = gmmfk.m(j) + np.matmul(K,innov1)
            Pc = gmmfk.P(j)  - multi_dot([K, Pz, K.T])

            Pu = alphas[j,0]*gmmfk.P(j)+(1-alphas[j,0])*Pc + multi_dot([K,innov2 - np.outer(innov1,innov1),K.T])

            gmmu.updateComp(j,m=xu,P=Pu)

        modelprobu=modelprobu/np.sum(modelprobu)

        return gmmu, modelprobu, gmmz, Pxzf,None



    @staticmethod
    def getEst(gmmfk,modelprobfk):
        m,P=gmmfk.weightedest(modelprobfk)
        return m,P


class TargetIMM:
    """
    IMM filter for a target

    """
    filterName = 'Target IMM Filter'

    def __init__(self,modefilterer):
        self.modefilterer = modefilterer
        self.immfilterer=IMMfilterer(self.modefilterer)

    def propagate(self,t, dt,target,uk, updttarget=True,**params):
        """
        propagate from t to t+dt
        so the final state is at t+dt at the end of this method
        """


        gmmfk, modelprobfk1 = self.immfilterer.propagate(t, dt,target.dynModelset,target.gmmfk,target.modelprobfk,uk,**params)

        if updttarget:
            target.setTargetFilterStageAsPrior()
            xfk,Pfk = self.immfilterer.getEst(gmmfk,modelprobfk1)
            target.updateParams(currt=t+dt,gmmfk=gmmfk,modelprobfk=modelprobfk1,xfk=xfk,Pfk=Pfk)

        return gmmfk, modelprobfk1


    def getPDFz(self,t, dt,target,sensormodel,cacheIntermediates2Target = True,**params):
        """

        """

#        pdfz =self.immfilterer.getPDFz(t, dt,target.gmmfk,target.modelprobfk,sensormodel)

        gmmu, modelprobu, gmmz,Pxzf, _ = self.immfilterer.measUpdate(t, dt,target.gmmfk,target.modelprobfk,sensormodel,None,inplace=False,**params)

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
        gmmu, modelprobu, gmmz,Pxzf, Lj  = self.immfilterer.measUpdate(t, dt,target.gmmfk,target.modelprobfk,sensormodel,zk,inplace=inplace,**params)

        if updttarget:
            target.setTargetFilterStageAsPosterior()
            xfu,Pfu = self.immfilterer.getEst(gmmu,modelprobu)
            target.updateParams(currt=t,gmmfk=gmmu,modelprobfk=modelprobu,xfk=xfu,Pfk=Pfu)

        return gmmu, modelprobu, gmmz,Pxzf


    def measWeightedBetasUpdt(self,t,dt,target,sensormodel, betas,Zk, updttarget=True, **params):
        """
        betas = [beta_0, beta_1, ...]
        Zk = [ zk1,zk2,...]
        beta_0 is for null measurement,i.e the target has no measurements
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
        gmmu, modelprobu, gmmz,Pxzf, _ = self.immfilterer.measWeightedBetasUpdt(t,dt,target.gmmfk,target.modelprobfk,sensormodel, betas,Zk, inplace=inplace, **params)

        if updttarget:
            target.setTargetFilterStageAsPosterior()
            xfu,Pfu = self.immfilterer.getEst(gmmu,modelprobu)
            target.updateParams(currt=t,gmmfk=gmmu,modelprobfk=modelprobu,xfk=xfu,Pfk=Pfu)



        return gmmu, modelprobu, gmmz,Pxzf

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
        self.immfilterer=IMMfilterer(self.modefilterer)

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
        xfk,Pfk = self.immfilterer.getEst(gmmfk,modelprobfk)

        self.groundtruthrecorder = uqrecorder.StatesRecorder_list(statetypes = ['xfk'] )

        if self.recordfilterstate:
            self.recorderprior.record(currt,gmmfk=self.gmmfk, modelprobfk=self.modelprobfk,xfk=xfk,Pfk=Pfk)
            self.recorderpost.record(currt,gmmfk=self.gmmfk, modelprobfk=self.modelprobfk,xfk=xfk,Pfk=Pfk)

    def setInitialSingledata(self,gmmfk,modelprobfk,currt):
        self.gmmfk = gmmfk
        self.currt = currt
        self.modelprobfk = modelprobfk
        xfk,Pfk = self.immfilterer.getEst(gmmfk,modelprobfk)
        if self.recordfilterstate:
            self.recorderprior.record(currt,gmmfk=self.gmmfk, modelprobfk=self.modelprobfk,xfk=xfk,Pfk=Pfk)
            self.recorderpost.record(currt,gmmfk=self.gmmfk, modelprobfk=self.modelprobfk,xfk=xfk,Pfk=Pfk)



    def propagate(self,t,dt,uk, **params):
        super().propagate(t, dt, uk, **params)


        gmmfk, modelprobfk = self.immfilterer.propagate(t, dt,self.dynMultiModels,self.gmmfk,self.modelprobfk,uk,**params)
        self.gmmfk = gmmfk
        self.modelprobfk = modelprobfk

        if self.recordfilterstate:
            xfk,Pfk = self.immfilterer.getEst(self.gmmfk,self.modelprobfk)
            self.recorderprior.record(t+dt,modelprobfk=self.modelprobfk,gmmfk=self.gmmfk,xfk = xfk,Pfk = Pfk)


    def measUpdate(self,t,dt, zk, **params):
        super().measUpdate(t,dt, zk, **params)

        gmmu, modelprobu, gmmz,Pxzf, Lj  = self.immfilterer.measUpdate(t, dt,self.gmmfk,self.modelprobfk,self.sensormodel,zk,inplace=True,**params)
        self.gmmfk = gmmu
        self.modelprobfk = modelprobu

        if self.recordfilterstate:
            xfk,Pfk = self.immfilterer.getEst(self.gmmfk,self.modelprobfk)
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

