# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import numpy as np
from uq.uqutils import recorder as uqrecorder
import copy
import uuid

import pdb
import matplotlib
# try:
#     matplotlib.use('TkAgg')
# except:
#     matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import collections as clc
import utils.plotting.geometryshapes as utpltgeom

from enum import Enum, auto


# active or deactive is to remove the target from computations and save resources
class RobotStatus(Enum):
    Active = auto()
    Inactive = auto()
    
class _baseRobot:
    def __init__(self):
        self.ID = uuid.uuid4()

class Robot2DRegGrid(_baseRobot):
    def __init__(self,mapobj=None):
        super().__init__()
        self.robotName = 1
        self.status = RobotStatus.Active
        self.mapobj = mapobj
        self.xk = np.array([np.nan,np.nan,np.nan]) # x,y,th
        self.shape={'a':5,'w':3}
        self.robotColor = 'r'
        self.robotAlpha = 0.4
        self.sensormodel = None
        self.dynModel = None
        self.recorder = uqrecorder.StatesRecorder_list(statetypes = ['xk','uk'] )
        self.ax=None
        self.controltemplates={}
        self.currt = 0
        self.controllerhistory={}
        self.statehistory={}
        
        self.NominalVmag = None
        self.MinTempXferr = None
        
        self.x0=None # some intiial time from which states are propagated 
        self.t0=None # some intiial time from which states are propagated
    def makeInActive(self):
        self.status = RobotStatus.Inactive()
    def makeActive(self):
        self.status = RobotStatus.Active()
    def isActive(self):
        return self.status == RobotStatus.Active()
        
        
    def freezeState(self):
        self.currt_freeze = np.copy(self.currt).astype(float)
        self.xk_freeze = self.xk.copy().astype(float)

    def defrostState(self):
        self.currt = np.copy(self.currt_freeze).astype(float)
        self.xk = self.xk_freeze.copy().astype(float)
        
        
    def setinitialState(self):
        self.x0 = self.xk
        self.t0 = self.currt
        
    def resetIntialState(self):
        self.xk = self.x0
        self.currt = self.t0
        
    def updateTime(self,currt):
        self.currt = currt
        
    def updateSensorModel(self,**params):
        params['xc'] = self.xk[0:2]
        params['dirn'] = self.xk[2]
        
        
        self.sensormodel.updateparam( self.currt, **params)
    
    def reachableNodesFrom(self,xk,T,returnNodeIds=False):
        radius = self.dynModel.maxvmag*T
        nodes = self.mapobj.nodesInRadius(xk[0:2],radius,returnNodeIds=returnNodeIds)
        return nodes
    
        
    def makeNewRobotfromCopy(self):
        robot = copy.deepcopy(self)
        robot.ID = uuid.uuid4()
        
        return robot
    
    def addTemplateTraj(self,uk_key=None,val=None):
        """
        key=(idx0,idth0,idxf,idthf)
        val={'xfpos':,'thf':,'Xtraj':,'cost':}

        """
        if uk_key in self.controltemplates:
            print("uk_key already in: ",uk_key )
        else:
            self.controltemplates[uk_key]=val
    
    def iterateControls(self,xnode0,th0):
        idx0 = self.mapobj.getNodeIdx(xnode0)
        idth0 = self.mapobj.getNodeDirnIdx(xnode0,th0)
        for idxf,xfpos in self.mapobj.iteratenodes():
            for idthf,thf in self.mapobj.iteratedirn(xfpos):
                key = (idx0,idth0,idxf,idthf)
                if key in self.controltemplates:
                    yield self.controltemplates[key]
                # else:
                #     print("key not in: ",key)
    
    def iterateControlsKeys(self,xnode0,th0):
        idx0 = self.mapobj.getNodeIdx(xnode0)
        idth0 = self.mapobj.getNodeDirnIdx(xnode0,th0)
        for idxf,xfpos in self.mapobj.iteratenodes():
            for idthf,thf in self.mapobj.iteratedirn(xfpos):
                key = (idx0,idth0,idxf,idthf)
                if key in self.controltemplates:
                    yield key
                # else:
                #     print("key not in: ",key)
                    
    def gettemplate(self,uk_key):
        return self.controltemplates.get(uk_key,None)
    
    def gettemplateCost(self,uk_key):
        contrval = self.controltemplates.get(uk_key,None)
        if contrval is None:
            return None
        else:
            return contrval['cost']
    
    def getcontrol(self,t):
        if t in self.controllerhistory:
            return self.controllerhistory[t]
        else:
            return None
    
    def propforward(self, t, dt, uk, **params):
        # first get current state and see if it is compatible with control
        x0idx = self.mapobj.getNodeIdx(self.xk[0:2])
        th0idx = self.mapobj.getNodeDirnIdx(self.xk[0:2],self.xk[2])
        if uk[0] == x0idx and uk[1] == th0idx:
            xfidx = uk[2]
            thfidx = uk[3]
            xf = self.mapobj.getNodefromIdx(xfidx)
            dirn = self.mapobj.getthfromIdx(thfidx)
            self.xk = np.hstack([xf,dirn])
            self.currt = t+dt
        else:
            raise LookupError("control ", uk," is not valid for current state ",x0idx," ",th0idx)
            
        
    def plotrobot(self,ax):
        self.sensormodel.plotsensorFOV(ax)
        
        triag = utpltgeom.getIsoTriangle(self.xk[0:2],self.shape['a'],self.shape['w'],self.xk[2])
        ax.plot(triag[:,0],triag[:,1],self.robotColor,alpha=self.robotAlpha,linewidth=2)
        
        ax.plot([self.xk[0]],[self.xk[1]],c=self.robotColor,marker='o',linewidth=2)
        ax.annotate(self.robotName,self.xk[0:2],self.xk[0:2]+2,color=self.robotColor)
        
    def plotrobotTraj(self,ax,tvec):
        """
        Plot robot trajectory from 0 to time t
        """
        for t in tvec:
            uk_key = self.getcontrol(t)
            if uk_key is not None:
                contval = self.gettemplate(uk_key)
                if contval is None:
                    print("--- control traj ------")
                    print(t,uk_key,contval)
                ax.plot(contval['Xtraj'][:,0],contval['Xtraj'][:,1],self.robotColor,linewidth=2 )
            

        
    def plotdebugTemplate(self,uk_key):
        contval = self.gettemplate(uk_key)
        
        fig = plt.figure(str(uk_key))
        ax = fig.add_subplot(111)
        
        # self.ax.cla()    
        self.mapobj.plotmap(ax)
        ax.plot(contval['Xtraj'][:,0],contval['Xtraj'][:,1],'r',linewidth=2 )
        M = np.min(contval['Xtraj'],axis=0)
        ax.set_xlim(M[0]-20, M[0]+20)
        ax.set_ylim(M[1]-20, M[1]+20)
        
        plt.pause(0.1)

if __name__=="__main__":
    robot = Robot2DRegGrid()
    robot.xk=np.array([5,5,np.pi/2])


    fig = plt.figure()
    ax = fig.add_subplot(111)
    robot.plotstate(ax)
    ax.axis('equal')
    plt.show()
