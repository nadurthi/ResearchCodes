#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

import logging
import numpy as np
import uuid
import copy
import numpy.linalg as nplg
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
from uq.uqutils import pdfs as uqutlpdfs
from uq.stats import pdfs as uqstatpdfs
from uq.stats import moments as uqstatmom
from utils.plotting import geometryshapes as utpltshp
from uq.quadratures import cubatures as uqcub

import scipy.linalg as sclnalg #block_diag
import scipy.optimize as scopt #block_diag
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class GMM:
    """
    All components have same dim
    """
    def __init__(self,ms,Ps,ws,t):
        self.ID = uuid.uuid4()
        self.mu = ms
        self.covars = Ps
        self.wts = ws
        self.currt = t
    
    def getCompPDF(self,i,asScipy=True):
        if asScipy is True:
            return multivariate_normal(self.mu[i],self.covars[i,:,:])
        else:
            return uqstatpdfs.GaussianPDF(self.mu[i],self.covars[i,:,:])
        
    def resetwts(self):
        self.wts=np.ones(self.Ncomp)/self.Ncomp
    
    def getwts(self):
        return self.wts.copy()
    
    def setwts(self,w):
        self.wts= w.copy().astype(float)
    
    def appendGMM(self,gmm):
        gmmC = gmm.makeCopy()
        
        if self.mu is None:
            self.mu = gmmC.mu.astype(float)
        else:
            self.mu = np.concatenate([self.mu,gmmC.mu],axis=0)
            
        if self.covars is None:
            self.covars = gmmC.covars.astype(float)
        else:
            self.covars = np.concatenate([self.covars,gmmC.covars],axis=0).astype(float)
            
        if self.wts is None:
            self.wts = gmmC.wts.astype(float)
        else:
            self.wts = np.concatenate([self.wts,gmmC.wts],axis=0).astype(float)
            
        self.currt = float(gmmC.currt)
    
    def collapseGMM(self):
        """
        return a GMM with 1 component        

        """        
        m,P=self.meanCov()
        return GMM.fromarray1Comp(m,P,self.currt)
    
    @classmethod
    def fromarray1Comp(cls,xf,Pf,t):
        return cls(np.stack([xf],axis=0).astype(float),np.stack([Pf],axis=0).astype(float),np.array([1]).astype(float),float(t))
    

    @classmethod
    def fromlist(cls,xf,Pf,wts,t):
        return cls(np.stack(xf,axis=0).astype(float),np.stack(Pf,axis=0).astype(float),np.array(wts).astype(float),float(t))

    def makeCopy(self):
        return copy.deepcopy(self)

    @property
    def dim(self):
        return len(self.mu[0])

    @property
    def Ncomp(self):
        return self.mu.shape[0]

    @property
    def idxs(self):
        return list(range(self.Ncomp))
    
    def updatetime(self,t):
        self.currt = float(t)

    def m(self,idx):
        return self.mu[idx]
    def P(self,idx):
        return self.covars[idx]
    def w(self,idx):
        return self.wts[idx]
    
    def scaleWt(self,c):
        self.wts = self.wts*c
        
    @property
    def mean(self):
        m,P=self.meanCov()
        return m
    
    @property
    def cov(self):
        m,P=self.meanCov()
        return P
    
    def __getitem__(self,idx):
        return self.mu[idx],self.covars[idx],self.wtss[idx]

    def updateComp(self,idx,**kwargs):
        if 'm' in kwargs:
            self.mu[idx] = kwargs['m'].astype(float)

        if 'P' in kwargs:
            self.covars[idx] = kwargs['P'].astype(float)

        if 'w' in kwargs:
            self.wts[idx] = kwargs['w'].astype(float)

    def deleteComp(self,idx,renormalize = True):
        self.mu = np.delete(self.mu,idx,axis=0)
        self.covars = np.delete(self.covars,idx,axis=0)
        self.wts = np.delete(self.wts,idx,axis=0)
       
        if renormalize is True:
            self.normalizeWts()
    
    def appendComp(self,m,P,w,renormalize = True):
        self.appendCompList([m.copy().astype(float)],[P.copy().astype(float)],[np.copy(w).astype(float)],renormalize = renormalize)
    
    def appendCompArray(self,m,P,w,renormalize = True):

        if self.mu is None:
            self.mu = m.copy().astype(float)
            self.covars = P.copy().astype(float)
            self.wts = np.copy(w).astype(float)
        else:            
            self.mu = np.concatenate([self.mu,m],axis=0).astype(float)
            self.covars = np.concatenate([self.covars,P],axis=0).astype(float)
            self.wts = np.concatenate([self.wts,w],axis=0).astype(float)
        
        if renormalize is True:
            self.normalizeWts()
            
    def appendCompList(self,ms,Ps,ws,renormalize = True):
        m = np.stack(ms,axis=0).astype(float)
        P = np.stack(Ps,axis=0).astype(float)
        w = np.stack(ws,axis=0).astype(float)
        if self.mu is None:
            self.mu = m.astype(float)
            self.covars = P.astype(float)
            self.wts = w.astype(float)
        else:
            self.mu = np.concatenate([self.mu,m],axis=0).astype(float)
            self.covars = np.concatenate([self.covars,P],axis=0).astype(float)
            self.wts = np.concatenate([self.wts,w],axis=0).astype(float)
            
        if renormalize is True:
            self.normalizeWts()
            
            
    def normalizeWts(self):
        self.wts = self.wts/np.sum(self.wts)

    def meanCov(self):
        return self.weightedest(self.wts)
    
    def evalcomp(self,x,idxs):
        """
        - if idxs is a vector, then return array evaluting x for each idx
        - if x has multiple rows, and idxs is vector return a list, each elelment is for 1 idx
        """
        if x.ndim>1:
            pdfeval = np.zeros((len(idxs),x.shape[0] ) )
        else:
            pdfeval = np.zeros(len(idxs))

        for i,idx in enumerate(idxs):
            pdfeval[i] = multivariate_normal.pdf(x,mean=self.mu[idx],cov=self.covars[idx])

        return pdfeval

    def pdf(self,x):

        pdfeval = self.evalcomp(x,self.idxs)
        if x.ndim>1:
            return np.sum(np.multiply(pdfeval,self.wts.reshape(-1,1)),axis=0)
        else:
            return np.sum(np.multiply(pdfeval,self.wts),axis=0)
        
    def isInNsig(self,x,N,compwise=True):
        flgs = []
        for idx in self.idxs:
            flgs.append( uqutlpdfs.isInNsig(x,self.mu[idx],self.covars[idx],N) )
        if compwise:
            return any(flgs)
        else:
            raise NotImplementedError()

    def marginalize(self,margdims,inplace=False):
        # margdims are dims to be removed
        sts = list(set(range(self.dim))-set(margdims))
        sts = sorted(sts);
#        % sts are the states to be kept others are removed

#        % define a new Gaussian mixture with only sts as first two states in order
        if inplace is False:
            return GMM(self.mu[:,sts],self.covars[np.ix_(range(self.Ncomp),sts,sts)],self.wts,self.currt)
        else:
            for idx in range(self.Ncomp):
                self = GMM(self.mu[:,sts],self.covars[np.ix_(range(self.Ncomp),sts,sts)],self.wts,self.currt)



    def weightedMerge(self,idxs,wts,inplaceidx=None):
        m = 0
        P = 0
        for i,idx in  enumerate(idxs):
            m = m + wts[i]*self.mu[idx]

        for i,idx in  enumerate(idxs):
            P = P + wts[i]*( self.covars[idx] + np.outer(self.mu[idx]-m,self.mu[idx]-m) )

        if inplaceidx is None:
            return m,P
        else:
            self.mu[inplaceidx] = m
            self.covars[inplaceidx] = P


    def weightedest(self,wts):
        m = 0
        P = 0
        for i,idx in  enumerate(self.idxs):
            m = m + wts[i]*self.mu[idx]

        for i,idx in  enumerate(self.idxs):
            P = P + wts[i]*( self.covars[idx] + np.outer(self.mu[idx]-m,self.mu[idx]-m) )

        return m,P



    def mergComp(idx1,idx2):
        pass
    
    def pruneByWts(self,wtthresh=1e-3,renormalize = True):
        indx = self.wts >= wtthresh
        self.wts = self.wts[indx]
        self.mu = self.mu[indx]
        self.covars = self.covars[indx]
        if renormalize is True:
            self.normalizeWts()
            
    def randomEqualWts(self,N):
        X=[]
        for i in range(self.Ncomp):
            x=np.random.multivariate_normal(self.mu[i],self.covars[i,:,:], int(np.round(self.wts[i]*N))+len(self.mu[0]) )
            X.append(x)
        
        return np.vstack(X)
    
    def random(self,N):
        X=[]
        W=np.cumsum(np.hstack([0,self.wts]))
        C = np.random.rand(N)
        counts,bins = np.histogram(C, bins=W)
        for i in range(self.Ncomp):
            x=np.random.multivariate_normal(self.mu[i],self.covars[i,:,:], counts[i] )
            X.append(x)

        
        return np.vstack(X)
    
    def generateUTpts(self):
        X=[]
        W=[]
        for i in range(self.Ncomp):
            x,w = uqcub.UT_sigmapoints(self.mu[i],self.covars[i])
            X.append(x)
            W.append(self.wts[i]*w)
        return np.vstack(X),np.hstack(W)
    
    def affineTransform(self,b,A,inplace=False):
        if inplace is False:
            gmm=self.makeCopy()
        else:
            gmm = self
        for i in range(self.Ncomp):
            gmm.mu[i]=np.matmul(A,self.mu[i])+b
            gmm.covars[i] = multi_dot([A,self.covars[i],A.T])
        
        return gmm
        
def marginalizeGMM(gmm,margstates):
    """
    margstates are the states that remain

    Parameters
    ----------
    gmm : TYPE
        DESCRIPTION.
    margstates : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    m = []
    P = []
    w = []
    
    m = gmm.mu[:,margstates].tolist()
    w = gmm.wts.tolist()
    P = gmm.covars[np.ix_(range(gmm.Ncomp),margstates,margstates)].tolist()

    
    return GMM.fromlist(m,P,w,gmm.currt)


def optimizeWts(gmm,x,pt):
    w0 = gmm.wts.copy()
    pest = gmm.evalcomp(x,gmm.idxs).T
    # func = lambda w: 1/len(pt)*np.sum((pt-np.sum(np.multiply(w,pest.T),axis=1))**2)  + (np.sum(w)-1)**2
    func = lambda w: np.hstack([1/len(pt)*(pt-np.sum(np.multiply(w,pest),axis=1) ) , 10*(np.sum(w)-1),w])
    jac = lambda w: np.vstack([-1/len(pt)*pest,np.ones(len(w0)),np.identity(len(w0))]) 
    res = scopt.least_squares(func, w0,bounds=(0,1),jac=jac)
    w = res.x
    w = w/np.sum(w)
    # print(res)
    return w



def plotGMM2Dcontour(gmm,nsig=1,N=100,rettype='stacked'):
    X=[]
    for i in range(gmm.Ncomp):
        X.append( utpltshp.getCovEllipsePoints2D(gmm.m(i),gmm.P(i),nsig=nsig,N=N) )
    if rettype == 'stacked':
        return np.stack(X,axis=0)
    if rettype == 'list':
        return X

def plotGMM2Dsurf(gmm,Ng=50):
    m,P = gmm.meanCov()
    A = sclnalg.sqrtm(P) 
    xg = np.linspace(m[0]-4*A[0,0],m[0]+4*A[0,0],Ng)
    yg = np.linspace(m[1]-4*A[1,1],m[1]+4*A[1,1],Ng)
    xx,yy = np.meshgrid(xg,yg )
    X=np.hstack([xx.reshape(-1,1),yy.reshape(-1,1)])
    p = gmm.pdf(X)
    p = p.reshape(Ng,Ng)
    
    return xx,yy,p
    
    
def nonlinearTransform(gmm,f,sigmamethod = uqcub.SigmaMethod.UT):
    gmmf = gmm.makeCopy()
    for i in range(gmm.Ncomp):
        X,w = uqcub.GaussianSigmaPtsMethodsDict[sigmamethod](gmm.m(i),gmm.P(i))
        Y=np.zeros(X.shape)
        for j in range(X.shape[0]):
            Y[j,:]=f(X[j,:])
        m,P = uqstatmom.MeanCov(Y,w)
        gmmf.updateComp(i,m=m,P=P)
    
    return gmmf

# %%

def fokkerwtupdt():
   pass
#   NN=@(x,mu,P,D)1/sqrt((2*pi)^D*det(P))*exp(-0.5*(x-mu)'*(P\(x-mu))); %Gaussian pdf
#   %evaluating M
#   M=zeros(ng,ng);
#   N=zeros(ng,ng);
#   for i=1:1:ng
#       for j=1:1:ng
#
#           P=reshape(P1x(i,:)+P1x(j,:),nx,nx);
#           M(i,j)=NN(mu1x(i,:)',mu1x(j,:)',P,nx);
#
#           mu=mux(j,:)';
#           sqP=sqrtm(reshape(Px(j,:),nx,nx));
#           for k=1:1:length(W)
#               Xk=sqP*X(k,:)'+mu;
#               N(i,j)=N(i,j)+W(k)*NN(model.fx(Xk,model.para_dt(1)),mu1x(i,:)',reshape(P1x(i,:),nx,nx)+model.Q,nx);
#           end
#       end
#   end
#   f=N*wx;
#   Aeq=ones(1,ng);
#   beq=1;
#   lb=zeros(1,ng);
#   ub=ones(1,ng);
#    [w1x,ff,extflg] = quadprog(M,-f,[],[],Aeq,beq,lb,ub);
#    extflg
#    GMM.mu=mu1x;
#    GMM.P=P1x;
#    GMM.w=w1x;
#


# %%

def Expt_GMM(GMM,f):
    pass
    ng=len(GMM.w);
#    nx=size(GMM.mu,2);
#    % calculates the expectation of any function w.r.t a gaussian mixture
#    [X,W]=GH_pts(zeros(nx,1),eye(nx),2);
#    H=zeros(ng,1);
#    for i=1:1:ng
#    mu=GMM.mu(i,:)';
#    sqP=sqrtm(reshape(GMM.P(i,:),nx,nx));
#
#    for j=1:1:length(W)
#          Xj=sqP*X(j,:)'+mu;
#          H(i)=H(i)+W(j)*real(f(Xj));
#    end
#    end
#    ef=H'*GMM.w;
#    end
#
#


# %%




#function D=dist_GM(g,h)
#N=length(g.w);
#K=length(h.w);
#d=zeros(N,K);
#for i=1:1:N
#    for k=1:1:K
#        Pi=reshape(g.P(i,:),sqrt(length(g.P(i,:))),sqrt(length(g.P(i,:))));
#        mi=g.mu(i,:)';
#%         wi=g.w(i);
#
#        Pk=reshape(h.P(k,:),sqrt(length(h.P(k,:))),sqrt(length(h.P(k,:))));
#        mk=h.mu(k,:)';
#%         wk=h.w(k);
#
#        inPi=inv(Pi);
#        inPk=inv(Pk);
#        C=1/2*(mi'*inPi+mk'*inPk)*inv(inPi+inPk)*(inPi*mi+inPk*mk)-1/2*(mi'*inPi*mi+mk'*inPk*mk);
#        mc=inv(inPi+inPk)*(inPi*mi+inPk*mk);
#        inPc=(inPi+inPk)/2;
#        Pc=inv(inPc);
#        BC=sqrt(1/sqrt(det(2*pi*Pi))*1/sqrt(det(2*pi*Pk)))*exp(C/2)*sqrt(det(2*pi*Pc));
#        d(i,k)=abs(sqrt(1-BC));
#    end
#end
#A=[];
#B=[];
#for i=1:1:K
#    A=vertcat(A,[zeros(1,(i-1)*N),ones(1,N),zeros(1,N*K-N-(i-1)*N)]);
#    B=horzcat(B,[1,zeros(1,N-1)]);
#end
#A=vertcat(A,B);
#for i=1:1:N-1
#    B=circshift(B',1)';
#    A=vertcat(A,B);
#end
#options=optimset('Display','off');
#[x,D] = linprog(reshape(d,N*K,1),[],[],A,[h.w;g.w],zeros(N*K,1),[],[],options);
#end
#



