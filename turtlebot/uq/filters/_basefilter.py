# -*- coding: utf-8 -*-
"""
Documentation for this module.
basfilter
More details.
"""
import logging
import numpy as np
import abc

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from uq.uqutils import recorder as uqrecorder

class FiltererBase:
    __metaclass__ = abc.ABCMeta
    filterName = 'FiltererBase'
    @staticmethod
    def propagate(t, dt,dynModel,xfk,Pfk,uk,**params):
        raise NotImplementedError("This is the base class")

    @staticmethod
    def measUpdate(t, dt,xfk,Pfk,sensormodel,zk):
        raise NotImplementedError("This is the base class")


class IntegratedBaseFilterStateModel:
    __metaclass__ = abc.ABCMeta

    filterName = 'IntegratedBaseFilterStateModel'
    def debugStatus(self):
        ss=[]
        ss.append( ['filterName',self.filterName] )
        ss.append( ['currtk',self.currtk] )
        ss.append( ['recordfilterstate',self.recordfilterstate] )
        ss.append( ['filterstage',self.filterstage] )
        ss = ss + self.dynModel.debugStatus()
        ss = ss + self.sensModel.debugStatus()

        return ss

    def __init__(self, dynModel=None, sensModel=None, recordfilterstate=False,
                 recorderobjprior=None,recorderobjpost=None,currt=0,xfk=None,Pfk=None):
        self.dynModel = dynModel
        self.sensModel = sensModel


        self.recordfilterstate = recordfilterstate


        self.xfk = xfk
        self.Pfk = Pfk
        self.currt = currt

        if recorderobjprior is None:
            self.recorderprior = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk'] )
        else:
            self.recorderprior = recorderobjprior

        if recorderobjpost is None:
            self.recorderpost = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk'] )
        else:
            self.recorderpost = recorderobjpost

    def setInitialFilterState(self, xf0, Pf0):
        self.xf0 = xf0
        self.Pf0 = Pf0

        self.xfk = xf0
        self.Pfk = Pf0

        if self.recordfilterstate:
            self.recordHistory()

    def recordHistory(self, *args, **kwargs):

        raise NotImplementedError('recordHistory has otbe written for derived class')

    def propagate(self, t,dt,uk, **params):
        self.currt = t + dt
#        self.filterstage = 'Time Updated'

    @abc.abstractmethod
    def Pz(self):
        pass
    def Pxz(self):
        pass
    def K(self):
        pass

    def measUpdate(self,t,dt, zk, **params):
#        self.filterstage = 'Measurement Updated'
        self.currt = t

    def measWeightedUpdt(self,t,dt, betas,Zk, *args, **kwargs):
        pass


    def pseudoMeasUpdt(self, *args, **kwargs):
        pass

    def getMeanCov(self, *args, **kwargs):
        return (self.xfk, self.Pfk)
