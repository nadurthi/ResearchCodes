import numpy as np
import numpy.linalg as nplnalg
import scipy.linalg as sclnalg
from uq.gmm import gmmbase as uqgmmbase
from uq.quadratures import cubatures as uqcub
import scipy.linalg as sclnalg #block_diag
import scipy.optimize as scopt #block_diag
from utils.math import geometry as utmthgeom
from scipy.stats import multivariate_normal
from sklearn import mixture
from uq.stats import pdfs as uqstpdf
import os
import pandas as pd
import pickle as pkl

class SplitterConfig:
    def __init__(self,**params):
        self.updateParams(**params)
        
    def updateParams(self,**params):
        self.alphaSpread = params.get('alphaSpread',2)
        self.Ngh = params.get('Ngh',5)
        self.nsig = params.get('nsig',1)
        self.Nmc = params.get('Nmc',1000)
        self.NcompIN = params.get('NcompIN',5)
        self.NcompOUT = params.get('NcompOUT',5)
        self.minfrac = params.get('minfrac',5)
        self.wtthreshprune=params.get('wtthreshprune',1e-4)
        
        self.sigL = params.get('sigL',0.2)
        self.wtL = params.get('wtL',0.2)
        self.rsfac = params.get('rsfac',1.2)
        
splitterConfigDefault = SplitterConfig()

def splitGaussianUT(m,P,w=1,splitterConfig=splitterConfigDefault):
    """
    Split the Gaussian PDF with mean m and cov P
    Parameters
    ----------
    m : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    n = len(m)
    beta = np.sqrt(1-(1/splitterConfig.alphaSpread))
    # w=1*(2*n*1/alpha+2*n*beta**2)
    wnew = 1/(2*n)
    
    X,wnew = uqcub.UT2n_sigmapoints(m, P)
    
    mnew = np.zeros((2*n,n))
    for i in range(X.shape[0]):
        mnew[i] = m + beta*(X[i,:]-m)
    
    Pnew = np.zeros((2*n,n,n))
    for i in range(X.shape[0]):
        Pnew[i] = (1/splitterConfig.alphaSpread) * P
    
    wnew = wnew * w
    return uqgmmbase.GMM(mnew,Pnew,wnew,0)




    

def split0I1Dlibrary(splitterConfig=splitterConfigDefault):
    sigL = splitterConfig.sigL
    wtL = splitterConfig.wtL
    a = splitterConfig.rsfac
    
    rs,w = uqcub.GH1Dpoints(0,1,splitterConfig.Ngh)
    idx = np.argsort(rs)
    rs = rs[idx]
    w = w[idx]
    
    rs = a*rs
    ind = rs>=0
    rshalf = rs[ind]
    whalf = w[ind]
    xgrid = np.linspace(-10,10,200)
    nc = len(rshalf) # w0 w1, w2  etc
    pt = uqstpdf.gaussianPDF1D(xgrid,0,1)
    
    # vars: first w then sigmas vars=[ w(nc), sig(nc) ]
    bounds=[]
    for i in range(nc):
        bounds.append((0,1))
    for i in range(nc):
        bounds.append((0,20))
        


    if rshalf[0]==0:
        A=np.hstack([1,2*np.ones(nc-1),np.zeros(nc)])
        unitypartconstraint = scopt.LinearConstraint(A, 0.99, 1.01)
        
        def func(x,rsfull,xgrid,pt):
            w0=x[0]
            nc=int(len(x)/2)
            wv = x[0:nc]
            sv = x[nc:]
            w = np.hstack([ wv[::-1],wv[1:]] )
            sig = np.hstack([ sv[::-1],sv[1:]] )

            pgrid =np.zeros(len(xgrid))
            for i in range(len(w)):
                pgrid=pgrid+w[i]*uqstpdf.gaussianPDF1D(xgrid,rsfull[i],sig[i]**2)
            
            return np.sum((pt-pgrid)**2)+ sigL*nplnalg.norm(np.diff(sig))+ wtL*nplnalg.norm(np.diff(w))
        
        x0 = np.hstack([0.2,0.8*np.ones(nc-1)/(2*(nc-1)),0.5*np.ones(nc) ])
        res = scopt.minimize(func, x0,args=(rs,xgrid,pt),constraints=[unitypartconstraint],bounds=bounds)
        
        x=res.x
        nc=int(len(x)/2)
        wv = x[0:nc]
        sv = x[nc:]
        w = np.hstack([ wv[::-1],wv[1:]] )
        sig = np.hstack([ sv[::-1],sv[1:]] )
        
    else:
        A=np.hstack([2*np.ones(nc),np.zeros(nc)])
        unitypartconstraint = scopt.LinearConstraint(A, 0.99, 1.01)
        
        def func(x,rsfull,xgrid,pt):
            nc=int(len(x)/2)
            wv = x[0:nc]
            sv = x[nc:]
            w = np.hstack([ wv[::-1],wv] )
            sig = np.hstack([ sv[::-1],sv] )

            pgrid =np.zeros(len(xgrid))
            for i in range(len(w)):
                pgrid=pgrid+w[i]*uqstpdf.gaussianPDF1D(xgrid,rsfull[i],sig[i]**2)
            
            return np.sum((pt-pgrid)**2)+ sigL*nplnalg.norm(np.diff(sig))+ wtL*nplnalg.norm(np.diff(w))
        
        x0 = np.hstack([0.5*np.ones(nc)/nc,0.5*np.ones(nc) ])
        res = scopt.minimize(func, x0,args=(rs,xgrid,pt), constraints=[unitypartconstraint],bounds=bounds)
    
        x=res.x
        nc=int(len(x)/2)
        wv = x[0:nc]
        sv = x[nc:]
        w = np.hstack([ wv[::-1],wv] )
        sig = np.hstack([ sv[::-1],sv] )
        
    w = w/np.sum(w)
    print("w= ",w)
    print("sig= ",sig)
    return rs,w,sig,xgrid,pt

def splitGaussianND(Mean,Covar,splitterConfig=splitterConfigDefault):
    nd = len(Mean)
    rs,w,sig,xgrid,pt = split0I1Dlibrary(splitterConfig=splitterConfig)
    
    
    x = [rs]*nd
    Xm = np.meshgrid(*x) 
    X = []
    for i in range(nd):
        X.append( Xm[i].reshape(-1) )
    M = np.vstack(X).T
    
    x = [sig**2]*nd
    Xm = np.meshgrid(*x)   
    X = []
    for i in range(nd):
        X.append( Xm[i].reshape(-1) )
    PP = np.vstack(X).T
    P=[]
    for i in range(PP.shape[0]):
        P.append( np.diag(PP[i,:]) )
    P = np.stack(P,axis=0)
    
    x = [w]*nd
    Xm = np.meshgrid(*x)   
    X = []
    for i in range(nd):
        X.append( Xm[i].reshape(-1) )
    W = np.vstack(X).T
    
    W = np.prod(W,axis=1)
    
    gmm = uqgmmbase.GMM(M,P,W,0)
    gmm.affineTransform(Mean,sclnalg.sqrtm(Covar),inplace=True)
    return gmm

    
def splitGaussian1D_ryanruss(N,ruleoption):
    filepath = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(filepath ,'GaussianSplitLib1D_ryanruss.pkl' )
    with open(filename,'rb') as F:
        D = pkl.load(F)
        
    rs = np.array(D['rule'+str(ruleoption)][N]['means'])
    w = np.array(D['rule'+str(ruleoption)][N]['wts'])
    sig = np.array(D['rule'+str(ruleoption)][N]['stds'])
    
    return rs,w,sig

def splitGaussianND_principleAxis_ryanruss(avec,m,P,N,ruleoption):
    rs,w,sig = splitGaussian1D_ryanruss(N,ruleoption)
    S=sclnalg.sqrtm(P)
    avecnorm = avec/nplnalg.norm(avec)
    th = 10*np.pi/180
    k=None
    for i in range(len(m)):
        a1 = S[:,i]/nplnalg.norm(S[:,i])
        if np.arccos(np.dot(avecnorm,a1))< th:
            k=i
    
    M=[]
    P=[]
    W=[]
    if k is not None:
        for i in range(len(w)):
            W.append(w[i])
            M.append(m+rs[i]*S[:,k]  )
            Sk = S.copy()
            Sk[:,k]=sig[i]*Sk[:,k]
            P.append(Sk*Sk.T)
    else:
        avecstar = np.matmul(nplnalg.inv(S),avec)
        avecstar = avecstar/nplnalg.norm(avecstar)
        aa=np.outer(avecstar,avecstar)
        for i in range(len(w)):
            W.append(w[i])
            M.append(m+rs[i]*np.matmul(S,avecstar)  )
            Sk = S.copy()
            Sk[:,k]=sig[i]*Sk[:,k]
            P.append(  np.matmul(S,np.matmul(np.identity(len(m))+(sig[i]**2-1)*aa,S.T))  )
            
    gmm = uqgmmbase.GMM.fromlist(M,P,W,0)
    return gmm

if __name__ == "__main__":

    splitterConfigDefault.updateParams(Ngh=3,rsfac=2,sigL=0.20,wtL=0.20)
    splitterConfig = splitterConfigDefault

    rs,w,sig,xgrid,pt = split0I1Dlibrary(splitterConfig=splitterConfig)
     
    print("w= ",w)
    print("sig= ",sig)
    pgrid =np.zeros(len(xgrid))
    individcomp = []
    for i in range(len(w)):
        dd = uqstpdf.gaussianPDF1D(xgrid,rs[i],sig[i]**2)
        individcomp.append(dd)
        pgrid=pgrid+w[i]*dd
        
    import matplotlib.pyplot as plt
    plt.close("all")
    fig = plt.figure("split1")
    ax = fig.add_subplot(111,label='contour')
    
    plt.plot(xgrid,pt,'bo')
    plt.plot(xgrid,pgrid,'r')
    
    fig = plt.figure("split2")
    ax = fig.add_subplot(111,label='contour')
    for cc in range(len(individcomp)):
        plt.plot(xgrid,individcomp[cc],'k')
    
    m = np.array([1,1])
    P = np.array([[1,0.5],[0.5,1]])
    gmm = splitGaussianND(m,P,splitterConfig=splitterConfigDefault)
    
    fig = plt.figure("split2Dgmm")
    ax = fig.add_subplot(111,label='contour')
    
    XX = uqgmmbase.plotGMM2Dcontour(gmm,nsig=1,N=100,rettype='list')
    for cc in range(gmm.Ncomp):
        ax.plot(XX[cc][:,0],XX[cc][:,1])

