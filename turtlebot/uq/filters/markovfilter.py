from uq.quadratures import cubatures as quadcub

from uq.filters._basefilter import IntegratedBaseFilterStateModel, FiltererBase
import numpy.linalg as nplg
import logging
import numpy as np
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
import pdb

from uq.uqutils import pdfs as uqutlpdfs
from uq.stats import moments as uqstmom

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




class TargetMarkovF:
    """
    Markov state filter for a target

    """
    filterName = 'Markov discrete Filter'

    def __init__(self):
        pass

    def propagate(self,t,dt,target,updttarget=True, **params):
        """
        propagate from t to t+dt
        so the final state is at t+dt at the end of this method
        """
        pk = target.xfk
        pkm1 = 1-pk
        p = np.vstack([pk,pkm1])
        P = target.dynModel.P
        pn = P.dot(p)
        xfk1 = pn[0]
    

        if updttarget:
            target.setTargetFilterStageAsPrior()
            target.updateParams(currt=t+dt,xfk=xfk1)

        return xfk1,None

    
    
    def measUpdate(self, t, dt, target, sensormodel,zk, updttarget=True, **params):
        """
        t is the time at which measurement update happens
        dt is just used in functions to compute something
        Eg: TP=0.8, TN=0.8, FP=0.2, FN=0.2
        """
        xfk = np.vstack([target.xfk,1-target.xfk]).T
        xfk1 = xfk.copy()
        
        xk = target.dynModel.X
        zpk,_,_ = sensormodel.detection( t, dt, xk)
        
        if zk is None:
            zk = zpk.copy()
        

        
        #  assume zk is truth
        for i in range(len(zpk)):
            if zk[i]==1 and zpk[i]==1:
                # targ inside and pred is inside  (TP)
                xfk1[i,0] = xfk[i,0]*sensormodel.TP
                xfk1[i,1] = xfk[i,1]*sensormodel.FP
            # elif zk[i]==1 and zpk[i]==0:
            #     # targ inside and pred is outside  (FN)
            #     xfk1[i,0] = xfk[i,0]*sensormodel.FN
            #     xfk1[i,1] = xfk[i,1]*sensormodel.TN
            # elif zk[i]==0 and zpk[i]==0:
            #     # targ outside and pred is outside  (TN)
            #     xfk1[i,0] = xfk[i,0]*sensormodel.TN
            #     xfk1[i,1] = xfk[i,1]*sensormodel.FN
            elif zk[i]==0 and zpk[i]==1:
                # targ outside and pred is inside  (FP)
                xfk1[i,0] = xfk[i,0]*sensormodel.FN
                xfk1[i,1] = xfk[i,1]*sensormodel.TN
        
        xfk1 = xfk1/np.sum(xfk1,axis=1).reshape(-1,1)
        xfk1 = xfk1[:,0]
        
        if updttarget:
            target.setTargetFilterStageAsPosterior()
            target.updateParams(currt=t,xfk=xfk1)

        return xfk1,None


