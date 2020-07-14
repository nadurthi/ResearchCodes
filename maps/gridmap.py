import logging
import numpy as np
import matplotlib
# try:
#     matplotlib.use('TkAgg')
# except:
#     matplotlib.use('Qt5Agg')
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
        self.xy0 = np.array(xy0)
        self.xyf = np.array(xyf)
        self.d = d
        self.xg = np.arange(xy0[0],xyf[0],d[0])
        self.yg = np.arange(xy0[1],xyf[1],d[1])

        self.Xgmesh,self.Ygmesh=np.meshgrid(self.xg, self.yg )
        self.XYgvec = np.hstack([ self.Xgmesh.reshape(-1,1),self.Ygmesh.reshape(-1,1) ])
        # the grid is from left to right
        # directions of movement in clockwise
        self.th=np.array([0,np.pi/2,np.pi,-np.pi/2])

        self.Ng = self.XYgvec.shape[0]
        self.adjnodes = np.zeros((self.Ng,4))
        self.nx = len(self.xg)
        self.ny = len(self.yg)
        
        
        self.recorder = uqrecorder.StatesRecorder_list(statetypes = ['xfk','Pfk'] )

        
        
        self.initiate()
    
    def middirn(self):
        return 0.5*(self.xy0+self.xyf)
    
    def isInMap(self,xk):
        if xk[0]< self.xy0[0] or xk[0]> self.xyf[0] or xk[1]< self.xy0[1] or xk[1]> self.xyf[1]:
            return False
        else:
            return True
        
    def initiate(self):
        for i in range(self.Ng):
            if self.XYgvec[i,1]+self.d[1]>self.xyf[1]:
                up = -1
            else:
                up = i+self.nx

            if self.XYgvec[i,1]-self.d[1]<self.xy0[1]:
                down = -1
            else:
                down = i - self.nx

            if self.XYgvec[i,0]+self.d[0]>self.xyf[0]:
                right = -1
            else:
                right = i+1

            if self.XYgvec[i,0]-self.d[0]<self.xy0[0]:
                left = -1
            else:
                left = i-1


            self.adjnodes[i,:] = [up,right,down,left]

        self.adjnodes = self.adjnodes.astype(int)

    def getadjNode(self,idx,dirnstr):
        nidx = self.adjnodes[idx][ self.dirns[dirnstr] ]
        return nidx, self.XYgvec[nidx]
    
    def getNodefromIdx(self,xidx):
        return self.XYgvec[xidx,:]
    
    def getthfromIdx(self,thidx):
        return self.th[thidx]
    
    def getNodeIdx(self,xnode):
        aa = np.where((self.XYgvec == xnode).all(axis=1))
        return aa[0][0]
    
    def getNodeDirnIdx(self,xnode,th):
        idx = self.getNodeIdx(xnode)
        aa=np.where(self.th==th)

        return aa[0][0]
    
    def iteratenodes(self):
        for i in range(self.XYgvec.shape[0]):
            yield i,self.XYgvec[i]

    def iteratedirn(self,xnode):
        
        for i in range(len(self.th)):
            yield i,self.th[i]
    
    
        
    def plotmap(self,ax):
        
        ax.plot(self.XYgvec[:,0],self.XYgvec[:,1],'b.',linewidth=1,markersize=1)

        ax.set_xlim(self.xy0[0]-self.d[0], self.xyf[0]+self.d[0])
        ax.set_ylim(self.xy0[1]-self.d[1], self.xyf[1]+self.d[1])
        
    
        
        

if __name__=="__main__":
    rgmap = Regular2DNodeGrid()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rgmap.plot(ax)
    plt.pause(0.1)
    
    