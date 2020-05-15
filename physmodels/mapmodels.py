import logging
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from uq.uqutils import recorder as uqrecorder

import uuid
import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class _baseMap:
    mapname = "base Map"
    def __init__(self):
        self.ID = uuid.uuid4()



class Regular2DNodeGrid(_baseMap):
    mapname = "Regular2DNodeGrid"
    def __init__(self,xy0=(0,0),xyf=(101,101),d=(10,10)):
        super().__init__()
        self.xy0 = xy0
        self.xyf = xyf
        self.d = d
        self.xg = np.arange(xy0[0],xyf[0],d[0])
        self.yg = np.arange(xy0[1],xyf[1],d[1])

        self.Xgmesh,self.Ygmesh=np.meshgrid(self.xg, self.yg )
        self.XYgvec = np.hstack([ self.Xgmesh.reshape(-1,1),self.Ygmesh.reshape(-1,1) ])
        # the grid is from left to right
        # directions of movement in clockwise
        self.dirns = {'up':0,'right':1,'down':2,'left':3}
        self.int2dirns = {0:'up',1:'right',2:'down',3:'left'}


        self.Ng = self.XYgvec.shape[0]
        self.adjnodes = np.zeros((self.Ng,4))
        self.nx = len(self.xg)
        self.ny = len(self.yg)

        self.recorder = uqrecorder.StatesRecorder_list(statetypes = {'xfk':(None,),'Pfk':(None,None)} )


        for i in range(self.Ng):
            if self.XYgvec[i,1]+d[1]>xyf[1]:
                up = -1
            else:
                up = i+self.nx

            if self.XYgvec[i,1]-d[1]<xy0[1]:
                down = -1
            else:
                down = i - self.nx

            if self.XYgvec[i,0]+d[0]>xyf[0]:
                right = -1
            else:
                right = i+1

            if self.XYgvec[i,0]-d[0]<xy0[0]:
                left = -1
            else:
                left = i-1


            self.adjnodes[i,:] = [up,right,down,left]

        self.adjnodes = self.adjnodes.astype(int)

    def getadjNode(self,idx,dirnstr):
        nidx = self.adjnodes[idx][ self.dirns[dirnstr] ]
        return nidx, self.XYgvec[nidx]

    def getReachNodesIdxs(self,idx):
        return self.adjnodes[idx]

    def getReachNodesIdxs_withPath2node(self,idx):
        L=[]
        for nidx in self.adjnodes[idx]:
            pathids = self.invarianttemplates(idx)
            for pp in pathids:
                L.append([nidx,pp])

        return L



    def invarianttemplates(self,idx):
        traj = None
        cost = None
        return traj,cost



    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(self.xy0[0], self.xyf[0])
        ax.set_ylim(self.xy0[1], self.xyf[1])

        for i in range(self.Ng):
            ax.cla()
            ax.plot(self.XYgvec[i,0],self.XYgvec[i,1],'bo')

            adjidx = self.adjnodes[i,:]
            print(i,adjidx)
            if adjidx[0]>=0:
                ax.plot(self.XYgvec[adjidx[0],0],self.XYgvec[adjidx[0],1],'rx')
            if adjidx[1]>=0:
                ax.plot(self.XYgvec[adjidx[1],0],self.XYgvec[adjidx[1],1],'gx')
            if adjidx[2]>=0:
                ax.plot(self.XYgvec[adjidx[2],0],self.XYgvec[adjidx[2],1],'kx')
            if adjidx[3]>=0:
                ax.plot(self.XYgvec[adjidx[3],0],self.XYgvec[adjidx[3],1],'mx')

            ax.set_xlim(self.xy0[0]-self.d[0], self.xyf[0]+self.d[0])
            ax.set_ylim(self.xy0[1]-self.d[1], self.xyf[1]+self.d[1])
            plt.pause(0.1)



if __name__=="__main__":
    rgmap = Regular2DNodeGrid()

    rgmap.plot()