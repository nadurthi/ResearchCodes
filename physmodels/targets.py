# -*- coding: utf-8 -*-

import logging
import numpy as np
import uuid
import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from uq.uqutils import recorder as uqrecorder
from uq.uqutils import helper as uqutilhelp
from uq.uqutils import constants as uqutilconst


from enum import Enum, auto



class TargetStatus(Enum):
    Active = auto()
    Deactive = auto()



class Target:
    """

        - track
        - ID
        - features
        - history of measurements associated
        - history of state (mean,cov)
        - model / models for propagation
        - initialize
        - death
        - occluded
        - detect target using some features
        - status
        - L2,L3,L4 context, etc information
        - ground truth
        - history of images+bounding box+ masks+ keypoints
        - car model, volume etc anyt other features
    """

    def __init__(self, dynModel=None,dynModelset=None,gmmfk=None, modelprob = None, xfk=None, Pfk=None, currt=0,recordfilterstate=False,
            status=TargetStatus.Active, recorderobjprior=None,recorderobjpost=None):

        self.ID = uuid.uuid4()


        self.dynModelset = dynModelset # multiple models are possible
        self.dynModel = dynModel

        self.gmmfk = gmmfk
        self.xfk = xfk
        self.Pfk = Pfk
        self.modelprob = modelprob
        self.currt = currt
        self.context ={}

        self.N = 100


        self.recordfilterstate = recordfilterstate

        self.filterstage = uqutilconst.FilterStage.Posterior
        self.status = status
        self.freezer = None


        if recorderobjprior is None:
            self.recorderprior = uqrecorder.StatesRecorder_list(statetypes = {'xfk':(None,),'Pfk':(None,None)} )
        else:
            self.recorderprior = recorderobjprior

        if recorderobjpost is None:
            self.recorderpost = uqrecorder.StatesRecorder_list(statetypes = {'xfk':(None,),'Pfk':(None,None)} )
        else:
            self.recorderpost = recorderobjpost

        self.groundtruthrecorder = None
        self.plottingrecorder = None


        if self.recordfilterstate:
            self.recorderprior.record(currt,xfk=self.xfk,Pfk=self.Pfk)
            self.recorderpost.record(currt,xfk=self.xfk,Pfk=self.Pfk)



    def setInitialdata(self,currt,xf0, Pf0):
        self.xfk = xf0
        self.Pfk = Pf0
        self.currt = currt
        if self.recordfilterstate:
            self.recorderprior.record(currt,xfk=self.xfk,Pfk=self.Pfk)
            self.recorderpost.record(currt,xfk=self.xfk,Pfk=self.Pfk)

    def setTargetFilterStageAsPrior(self):
        self.filterstage = uqutilconst.FilterStage.Prior

    def setTargetFilterStageAsPosterior(self):
        self.filterstage = uqutilconst.FilterStage.Posterior

    def updateParams(self,**params):
        for key,value in params.items():
            if hasattr(self,key):
                setattr(self,key,value)
            else:
                raise KeyError("Target does not have this key '%s' to update"%(key,))

        if self.recordfilterstate:

            if self.filterstage == uqutilconst.FilterStage.Prior:
                self.recorderprior.record(self.currt,xfk=self.xfk,Pfk=self.Pfk)

            if self.filterstage == uqutilconst.FilterStage.Posterior:
                self.recorderpost.record(self.currt,xfk=self.xfk,Pfk=self.Pfk)

    def freeze(self,recorderobj=False):
        self.freezer ={}
        self.freezer['dynModel'] = copy.deepcopy(self.dynModel)
        self.freezer['dynModelset'] = copy.deepcopy(self.dynModelset)
        self.freezer['xfk'] = copy.deepcopy(self.xfk)
        self.freezer['Pfk'] = copy.deepcopy(self.Pfk)
        self.freezer['gmmfk'] = copy.deepcopy(self.gmmfk)
        self.freezer['modelprob'] = copy.deepcopy(self.modelprob)
        self.freezer['currt'] = copy.deepcopy(self.currt)
        self.freezer['status'] = self.status
        self.freezer['filterstage'] = self.filterstage

        if recorderobj:
            self.freezer['recorderprior'] = self.recorderprior.makecopy()
            self.freezer['recorderpost'] = self.recorderpost.makecopy()
            self.freezer['groundtruthrecorder'] = copy.deepcopy(self.groundtruthrecorder)
            self.freezer['plottingrecorder'] = copy.deepcopy(self.plottingrecorder)

    def defrost(self,recorderobj=False):
        if self.freezer is None:
            raise KeyError("Freezer is empty")

        self.dynModel = copy.deepcopy(self.freezer['dynModel'])
        self.dynModelset = copy.deepcopy(self.freezer['dynModelset'])  # multiple models are possible
        self.xfk = copy.deepcopy(self.freezer['xfk'])
        self.Pfk = copy.deepcopy(self.freezer['Pfk'])
        self.currt = copy.deepcopy(self.freezer['currt'])
        self.gmmfk = copy.deepcopy(self.freezer['gmmfk'])
        self.modelprob = copy.deepcopy(self.freezer['modelprob'])
        self.status = self.freezer['status']
        self.filterstage = self.freezer['filterstage']

        if recorderobj:
            self.recorderprior = self.freezer['recorderprior'].makecopy()
            self.recorderpost = self.freezer['recorderpost'].makecopy()
            self.groundtruthrecorder = copy.deepcopy(self.freezer['groundtruthrecorder'])
            self.plottingrecorder = copy.deepcopy( self.freezer['plottingrecorder'] )


    def isactive(self):
        if self.status == TargetStatus.Active:
            return True
        else:
            return False

    def debugStatus(self):
        ss=[]
        ss.append( ['----------','----------'] )
        ss.append( ['Target',self.ID] )
        ss.append( ['----------','----------'] )


        ss.append( ['xfk',self.xfk] )

        ss.append( ['Pfk',self.Pfk] )
        ss.append( ['recordfilterstate',self.recordfilterstate] )
        ss.append( ['filterstage',self.filterstage] )
        ss.append( ['currt',self.currt] )
        ss.append( ['status',self.status] )
        ss.append( ['State Recorder',''] )
        ss = ss + [['\t'+s1[0],s1[1]] for s1 in self.recorderprior.debugStatus()]
        ss = ss + [['\t'+s1[0],s1[1]] for s1 in self.recorderpost.debugStatus()]

        if self.groundtruthrecorder is not None:
            ss.append( ['Ground Truth Recorder',''] )
            ss = ss + [['\t'+s1[0],s1[1]] for s1 in self.groundtruthrecorder.debugStatus()]

        ss = ss + [['\t'+s1[0],s1[1]] for s1 in self.dynModel.debugStatus()]

        return ss

    def __str__(self):
        debugstatuslist = self.debugStatus()
        ss = uqutilhelp.DebugStatus().returnstatus(debugstatuslist)
        return ss




class TargetSet:
    def __init__(self):
        self.targets = []
        self.ID = uuid.uuid4()



    def targetIDs(self):
        return [ss.ID for ss in self.targets]


    @property
    def ntargs(self):
        return len(self.targets)

    @property
    def ntargs_active(self):
        return len([x.ID for x in self.targets if x.isactive()])

    @property
    def ntargs_inactive(self):
        return len([x.ID for x in self.targets if not x.isactive()])

    def __getitem__(self,i):
        return self.targets[i]

    def mergeTargets(self, targetID1, targetID2):
        pass

    def addTarget(self, target):
        self.targets.append(target)

    def deleteTargets(self, targetIDs):
        self.targets = filter(lambda x: x.ID not in targetIDs, self.targets)

    def deleteInactiveTargets(self):
        self.targets = filter(lambda x: x.isactive(), self.targets)

    def debugStatus(self):
        ss = []
        ss.append( ['----------','----------'] )
        ss.append( ['Targetset',self.ID] )
        ss.append( ['----------','----------'] )
        ss.append( ['ntargs',self.ntargs] )
        ss.append( ['ntargs_active',self.ntargs_active]   )
        ss.append( ['ntargs_inactive',self.ntargs_inactive] )
        for i in range(self.ntargs):
            ss = ss + [['\t'+s1[0],s1[1]] for s1 in self.targets[i].debugStatus()]

        return ss