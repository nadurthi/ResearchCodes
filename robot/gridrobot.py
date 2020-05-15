# -*- coding: utf-8 -*-

import numpy as np
from uq.uqutils import recorder as uqrecorder
import logging
import uuid
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class _baseRobot:
    def __init__(self):
        self.ID = uuid.uuid4()

class Robot2DRegGridMap(_baseRobot):
    def __init__(self,mapobj=None):
        self.mapobj = mapobj
        self.xk = [None,None,None] # x,y,dirn
        self.headingdirections = {0:'up',1:'right',2:'down',3:'left'}
        self.reachradius = 11
        self.color = 'r'
        self.sensormodel = None

        self.recorder = uqrecorder.StatesRecorder_list(statetypes = {'xfk':(None,),'Pfk':(None,None)} )


    def setfov(self):
        self.rmax = 30
        self.fovshape = 'circular'

    def plotstate(self,ax):
        triag = np.array([[0,2],[1,-2],[-1,-2],[0,2]])
        if self.xk[2]==0:
            th=np.pi/2
        if self.xk[2]==1:
            th=0
        if self.xk[2]==2:
            th=-np.pi/2
        if self.xk[2]==3:
            th=np.pi

        R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        for i in range(triag.shape[0]):
            triag[i] = np.matmul(R,triag[i])

        triag[:,0]=triag[:,0]+self.xk[0]
        triag[:,1]=triag[:,1]+self.xk[1]


        ax.plot(triag[:,0],triag[:,1],self.color)
        ax.plot(self.xk[0],self.xk[1],self.color+'o')

        if self.fovshape == 'circular':
            Xfov=[]
            for th in np.linspace(0,2*np.pi,100):
                Xfov.append( [self.rmax*np.cos(th),self.rmax*np.sin(th)] )

            Xfov = np.array(Xfov)

            Xfov[:,0]=Xfov[:,0]+self.xk[0]
            Xfov[:,1]=Xfov[:,1]+self.xk[1]

        ax.fill(Xfov[:,0],Xfov[:,1],self.color,alpha=0.4,
                edgecolor=self.color,facecolor=self.color)

if __name__=="__main__":
    robot = Robot2DRegGridMap()
    robot.xk=[5,5,1]
    robot.setfov()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    robot.plotstate(ax)
    ax.axis('equal')
    plt.show()
