#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

import numpy as np
import pdb

class Metrics:
    def __init__(self):
        pass



def getEsterror(xt,xest,stateset=None):
    """
    stateset = {'pos':[0,1],'vel':[2,3],'x':[0]}
    """
    N,dim = xest.shape
    if stateset is None:
        stateset = {'state':np.arange(dim) }

    errt={}
    rmse={}
    for state in stateset:
        ssind = stateset[state]
        # pdb.set_trace()
        errt[state] = np.sqrt( np.sum(np.power(xt[:,ssind] - xest[:,ssind],2),axis=1) )
        rmse[state] = np.sqrt( np.sum(np.power(errt[state], 2))/N )

    return errt,rmse


def getEsterror2ClosestTarget(xtlist,xest,stateset=None):
    """
    stateset = {'pos':[0,1],'vel':[2,3],'x':[0]}
    """
    nt = len(xtlist)
    N,dim = xest.shape
    if stateset is None:
        stateset = {'state':np.arange(dim) }

    errt={}
    rmse={}
    
    for state in stateset:
        m1=1e15
        for xt in xtlist:        
            ssind = stateset[state]
            # pdb.set_trace()
            e1=np.sqrt( np.sum(np.power(xt[:,ssind] - xest[:,ssind],2),axis=1) )
            e2=np.sqrt( np.sum(np.power(e1, 2))/N )
            if e2<m1:
                m1=e2
                errt[state] = e1
                rmse[state] = e2

    return errt,rmse