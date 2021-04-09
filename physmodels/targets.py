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


from enum import Enum, auto, IntEnum


# active or deactive is to remove the target from computations and save resources
class TargetStatus(IntEnum):
    Active = auto()
    InActive = auto()

class TargetTrack(IntEnum):
    Maintained = auto()
    Lost = auto()
    Recovered = auto()

class TargetObservability(IntEnum):
    NotObservable = auto()
    Observable = auto()
    Occluded = auto()

class TargetType(IntEnum):
    Standard = auto()
    Search = auto()

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

    def __init__(self, dynModel=None,dynModelset=None,gmmfk=None, modelprobfk = None, xfk=None, Pfk=None,particlesfk=None, currt=0,recordfilterstate=False,
            status=TargetStatus.Active, recorderobjprior=None,recorderobjpost=None,color='k'):

        self.ID = uuid.uuid4()
        self.targetName = 1
        

        self.dynModelset = dynModelset # multiple models are possible
        self.dynModel = dynModel
        
        self.particlesfk = particlesfk
        self.gmmfk = gmmfk
        self.xfk = xfk
        self.Pfk = Pfk
        self.modelprobfk = modelprobfk
        
        self.gmmtk = None  # truth
        self.xtk = None  # truth
        self.Ptk = None  # truth
        self.modelprobtk = None  # truth
        self.posstates=np.array([0,1])
        
        self.tempData={}
        
        self.currt = currt
        self.context ={}
        
        self.color=color
        
        self.N = 100

        self.targtype = TargetType.Standard
        self.filterer = None

        self.recordfilterstate = recordfilterstate

        self.filterstage = uqutilconst.FilterStage.Posterior
        self.status = status
        self.freezer = None


        if recorderobjprior is None:
            self.recorderprior = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk'] )
        else:
            self.recorderprior = recorderobjprior

        if recorderobjpost is None:
            self.recorderpost = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk'] )
        else:
            self.recorderpost = recorderobjpost

        self.groundtruthrecorder = uqrecorder.StatesRecorder_list(statetypes = ['xtk'] )
        self.plottingrecorder = None

#
#        if self.recordfilterstate:
#            params={}
#            for recstate in self.recorderprior.states:
#                params[recstate] = getattr(self,recstate)
#            self.recorderprior.record(currt,**params)
#
#            for recstate in self.recorderpost.states:
#                params[recstate] = getattr(self,recstate)
#            self.recorderpost.record(currt,**params)
    
    def makeCopy(self):
        return copy.deepcopy(self)
    
    def isSearchTarget(self):
        return self.targtype == TargetType.Search
    
    def isActive(self):
        return self.status == TargetStatus.Active
    def isInActive(self):
        return self.status == TargetStatus.InActive
    
    def makeInactive(self):
        self.status = TargetStatus.InActive
    def makeActive(self):
        self.status = TargetStatus.Active
        
    def freezeState(self):
        raise NotImplemented("Deprecated")
        if self.gmmfk is not None:
            self.gmmfk_freezer = self.gmmfk.makeCopy()
        else:
            self.gmmfk_freezer = None
            
        if self.xfk is not None:
            self.xfk_freezer = self.xfk.copy()
        else:
            self.xfk_freezer = None
            
        if self.Pfk is not None:
            self.Pfk_freezer = self.Pfk.copy()
        else:
            self.Pfk_freezer = None
            
        if self.modelprobfk is not None:
            self.modelprobfk_freezer = self.modelprobfk.copy()
        else:
            self.modelprobfk_freezer = None
            
        if self.currt is not None:
            self.currt_freezer = np.copy(self.currt).astype(float)
        else:
            self.currt_freezer = None
            
    def defrostState(self):
        raise NotImplemented("Deprecated")
        if self.gmmfk_freezer is None:
            self.gmmfk = self.gmmfk_freezer
        else:
            self.gmmfk = self.gmmfk_freezer.makeCopy()
        
        if self.xfk_freezer is None:
            self.xfk = self.xfk_freezer
        else:
            self.xfk = self.xfk_freezer.copy()
        
        if self.Pfk_freezer is None:
            self.Pfk = self.Pfk_freezer
        else:
            self.Pfk = self.Pfk_freezer.copy()
        
        if self.modelprobfk_freezer is None:
            self.modelprobfk = self.modelprobfk_freezer
        else:
            self.modelprobfk = self.modelprobfk_freezer.copy()
        
        if self.currt_freezer is None:
            self.currt = self.currt_freezer
        else:
            self.currt = np.copy(self.currt_freezer).astype(float)
        
        
    def setStateFromPrior(self,t,statevars):
        raise NotImplemented("Deprecated")
        for var in statevars:
            v = self.recorderprior.getvar_bytime(var,t)
            if v is None:
                raise Exception("setStateFromPrior: ",var," not in recorderprior at time ",t)
            setattr(self,var,v)
        self.currt=t
        
    def setStateFromPost(self,t,statevars):
        raise NotImplemented("Deprecated")
        for var in statevars:
            v = self.recorderpost.getvar_bytime(var,t)
            if v is None:
                raise Exception("setStateFromPost: ",var," not in recorderpost at time ",t)
            setattr(self,var,v)
        self.currt=t
        
    def setInitialdata(self,currt,xfk=None, Pfk=None,gmmfk=None, modelprobfk=None,particlesfk=None):
        self.xfk = xfk
        self.Pfk = Pfk
        self.gmmfk = gmmfk
        self.modelprobfk = modelprobfk
        self.particlesfk = particlesfk
        
        self.currt = currt

        if self.recordfilterstate:
            params={}
            for recstate in self.recorderprior.states:
                if isinstance(getattr(self,recstate),IntEnum):
                    params[recstate] = getattr(self,recstate).name
                else:
                    params[recstate] = getattr(self,recstate)
                    
            self.recorderprior.record(currt,**params)

            params={}
            for recstate in self.recorderpost.states:
                if isinstance(getattr(self,recstate),IntEnum):
                    params[recstate] = getattr(self,recstate).name
                else:
                    params[recstate] = getattr(self,recstate)
                    
            self.recorderpost.record(currt,**params)

    def setTargetFilterStageAsPrior(self):
        self.filterstage = uqutilconst.FilterStage.Prior

    def setTargetFilterStageAsPosterior(self):
        self.filterstage = uqutilconst.FilterStage.Posterior
    
    
        
    def updateParams(self,**params):
        for key,value in params.items():
            if hasattr(self,key):
                if isinstance(getattr(self,key),TargetStatus):
                    for ss in TargetStatus:
                        if ss.name==value:
                            setattr(self,key,ss)        
                else:
                    setattr(self,key,value)
            else:
                raise KeyError("Target does not have this key '%s' to update"%(key,))

        if self.recordfilterstate:

            if self.filterstage == uqutilconst.FilterStage.Prior:
                recparams={}
                for recstate in self.recorderprior.states:
                    if isinstance(getattr(self,recstate),IntEnum):
                        recparams[recstate] = getattr(self,recstate).name
                    else:
                        recparams[recstate] = getattr(self,recstate)
                self.recorderprior.record(self.currt,**recparams)

            if self.filterstage == uqutilconst.FilterStage.Posterior:
                recparams={}
                for recstate in self.recorderpost.states:
                    if isinstance(getattr(self,recstate),IntEnum):
                        recparams[recstate] = getattr(self,recstate).name
                    else:
                        recparams[recstate] = getattr(self,recstate)
                self.recorderpost.record(self.currt,**recparams)
    
    def resetState2timePriorRecord(self,t):
        for var in self.recorderprior.data:
            if var=='t':
                continue
            if hasattr(self,var) :
                val = self.recorderprior.getvar_bytime(var,t)
                if isinstance(getattr(self,var),TargetStatus):
                    for ss in TargetStatus:
                        if ss.name==val:
                            setattr(self,var,ss) 
                else:
                    setattr(self,var,val)
                self.currt = t        
            else:
                print("Target does not have this key '%s' to update"%(var,))
        
    def resetState2timePostRecord(self,t):
        for var in self.recorderpost.data:
            if var=='t':
                continue
            if hasattr(self,var):
                val = self.recorderpost.getvar_bytime(var,t)
                if isinstance(getattr(self,var),TargetStatus):
                    for ss in TargetStatus:
                        if ss.name==val:
                            setattr(self,var,ss) 
                else:
                    setattr(self,var,val)
                self.currt = t
            else:
                print("Target does not have this key '%s' to update"%(var,))
                
                
    def freeze(self,recorderobj=False):
        self.freezer ={}
        self.freezer['dynModel'] = copy.deepcopy(self.dynModel)
        self.freezer['dynModelset'] = copy.deepcopy(self.dynModelset)
        self.freezer['xfk'] = copy.deepcopy(self.xfk)
        self.freezer['Pfk'] = copy.deepcopy(self.Pfk)
        self.freezer['gmmfk'] = copy.deepcopy(self.gmmfk)
        self.freezer['modelprobfk'] = copy.deepcopy(self.modelprobfk)
        self.freezer['currt'] = copy.deepcopy(self.currt)
        self.freezer['status'] = self.status
        self.freezer['filterstage'] = self.filterstage
        
        self.freeze_time = copy.deepcopy(self.currt)
        
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
        self.modelprobfk = copy.deepcopy(self.freezer['modelprobfk'])
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
        
    def hasTarget(self,ID):
        for i in range(self.ntargs):
            if self.targets[i].ID == ID:
                return True
        return False
    
    def getTargetByID(self,ID):
        for i in range(self.ntargs):
            if self.targets[i].ID == ID:
                return self.targets[i]
        return None
    
    def asserteqtimes(self):
        T=[]
        for i in range(self.ntargs):
            T.append(self.targets[i].currt)
        np.testing.assert_approx_equal(np.max(T),np.min(T))

    def targetIDs(self):
        return [ss.ID for ss in self.targets]

    
    @property
    def targsIDlist(self):
        return [targ.ID for targ in self.targets]
    
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