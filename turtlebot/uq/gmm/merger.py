# -*- coding: utf-8 -*-
import matplotlib
# try:
#     matplotlib.use('TkAgg')
# except:
#     matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nplng
from uq.gmm import gmmbase as uqgmmbase
from uq.information import distance
from utils.plotting import geometryshapes as utpltshp
import pdb

class MergerConfig:
    def __init__(self,**params):
        self.updateParams(**params)
        
    def updateParams(self,**params):
        self.meanabs = params.get('meanabs',0.5)
        self.meanthresfrac = params.get('meanthresfrac',0.5)
        self.dh = params.get('dh',0.5)
        self.wtthreshprune=params.get('wtthreshprune',1e-4)
        self.doMerge=params.get('doMerge',True)
        self.fixedComps=params.get('fixedComps',7)
        self.alorithm=params.get('alorithm','EMfixedComps')
        self.fixedCompsNmc=params.get('fixedCompsNmc',1000)
        
mergerConfigDefault = MergerConfig()

        
def mergeGaussians(g1,g2,w1=0.5,w2=0.5):
    w = w1 + w2

    m = 1/w*(w1*g1.mean+w2*g2.mean)
    P = w1*(g1.cov+np.outer(g1.mean-m,g1.mean-m)) + w2*(g2.cov+np.outer(g2.mean-m,g2.mean-m))
    P = P/w
        
    return m, P, w

def merge1(gmm,mergerConfig=mergerConfigDefault):
    meanabs = mergerConfig.meanabs
    meanthresfrac = mergerConfig.meanthresfrac
    dh = mergerConfig.dh
    
    # fig = plt.figure("merging-process")
    # ax = fig.add_subplot(111)
    # X2 = uqgmmbase.plotGMM2Dcontour(gmm,nsig=1,N=100,rettype='list')
    
        
    Dm = np.zeros((gmm.Ncomp,gmm.Ncomp))
    Dh = np.zeros((gmm.Ncomp,gmm.Ncomp))
    Dmfrac = np.zeros((gmm.Ncomp,gmm.Ncomp))
    Do = np.zeros((gmm.Ncomp,gmm.Ncomp))
    for i in range(gmm.Ncomp):
        for j in range(i,gmm.Ncomp):
            Dh[i,j] = distance.hellingerDist(gmm.getCompPDF(i),gmm.getCompPDF(j))
            Dm[i,j] = nplng.norm(gmm.m(i)-gmm.m(j))
            d = np.min(np.sqrt(np.hstack([np.diag(gmm.P(i)),np.diag(gmm.P(j))])))
            Dmfrac[i,j] = Dm[i,j]/d
            # eigval,eigvec = nplng.eig(gmm.P(i))
    
    # print(Dh)
    mnew = []
    Pnew = []
    wnew = []
    doneComp = []
    # first merge by closest mean and hellinger dist
    for i in range(gmm.Ncomp):
        if i in doneComp:
            continue
        k=-1
        mmm=100000
        for j in range(i+1,gmm.Ncomp):
            if j not in doneComp:
                if Dh[i,j]<mmm:
                    mmm=Dh[i,j]
                    k=j
        if k==-1:
            mnew.append(gmm.m(i))
            Pnew.append(gmm.P(i))
            wnew.append(gmm.w(i))
            doneComp.append(i) 
            continue
        
        # k=np.argmax(Dh[i,:])

        # print(Dh[i,k])
        if Dh[i,k]<dh:
            m, P, w= mergeGaussians(gmm.getCompPDF(i),gmm.getCompPDF(k),w1=gmm.w(i),w2=gmm.w(k))
            mnew.append(m)
            Pnew.append(P)
            wnew.append(w)
            doneComp.append(i)
            doneComp.append(k)
        else:
            mnew.append(gmm.m(i))
            Pnew.append(gmm.P(i))
            wnew.append(gmm.w(i))
            doneComp.append(i) 

    gmmnew = uqgmmbase.GMM.fromlist(mnew,Pnew,wnew,gmm.currt)
    gmmnew.normalizeWts()
    
    print("MERGING: Before: %d, After: %d"%(gmm.Ncomp,gmmnew.Ncomp))
    return gmmnew
