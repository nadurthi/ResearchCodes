import numpy as np
import numpy.linalg as nplg
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
import scipy.linalg as sclalg
import scipy.optimize as scopt

import warnings
import numba as nb


def WassersteinDist(m1,P1,m2,P2):
    s1 = sclalg.sqrtm(P1)
    g = np.matmul(np.matmul(s1,P2),s1) 
    d = np.sqrt( nplg.norm(m1-m2)+np.trace(  P1+P2-2*sclalg.sqrtm( g  ) ) )
    
    return d

def WassersteinDist_gaussian(g1,g2):
    m1=g1.mean 
    P1=g1.cov
    m2=g2.mean 
    P2=g2.cov
   
    return WassersteinDist(m1,P1,m2,P2)

def BCoeff_gassian(g1,g2):
    P1inv = nplg.inv(g1.cov)
    P2inv = nplg.inv(g2.cov)
    a = np.matmul(g1.mean,P1inv)+ np.matmul(g2.mean,P2inv)
    b = nplg.inv(P1inv+P2inv)
    
    d = np.dot(np.matmul(g1.mean,P1inv),g1.mean) + np.dot(np.matmul(g2.mean,P2inv),g2.mean)
    C = 0.5*np.dot(np.matmul(a,b),a)-0.5*d
    bc = np.sqrt( nplg.det(2*nplg.inv(P1inv+P2inv)) /np.sqrt(nplg.det(np.matmul(g1.cov,g2.cov))) ) * np.exp(C/2)
    return bc

def hellingerDist(g1,g2):
    if np.allclose(g1.mean, g2.mean, rtol=1e-05, atol=1e-08) and np.allclose(g1.cov, g2.cov, rtol=1e-05, atol=1e-08):
        BC=1
    else:
        BC=BCoeff_gassian(g1, g2)
        
    if BC>1:
        BC=1
    return np.sqrt(1-BC)

def KLDist(g1,g2):
    if np.allclose(g1.mean, g2.mean, rtol=1e-05, atol=1e-08) and np.allclose(g1.cov, g2.cov, rtol=1e-05, atol=1e-08):
        BC=1
    else:
        BC=BCoeff_gassian(g1, g2)
        
    if BC>1:
        BC=1
    return np.sqrt(1-BC)


def hellingerDistGMM(gmm1,gmm2):
    n1 = gmm1.Ncomp
    n2 = gmm2.Ncomp
    D=np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            D[i,j] = hellingerDist(gmm1.getCompPDF(i,asScipy=False),gmm2.getCompPDF(j,asScipy=False))
            
            
    A=[]
    for i in range(n1):
        z = np.zeros(n1*n2)
        z[i*n2:(i+1)*n2] = np.ones(n2)
        A.append(z)
    
    B=[]
    for i in range(n1):
        B.append(np.identity(n2))
    
    Aeq = np.vstack([A,np.hstack(B)])    
    Beq = np.hstack([gmm1.wts,gmm2.wts])
    
    # print(Aeq)
    # print(Beq)
    # print(D)
    warnings.filterwarnings("ignore")
    res = scopt.linprog(D.reshape(-1), A_ub=None, b_ub=None, A_eq=Aeq, b_eq=Beq)
    # print(res)
    warnings.filterwarnings("default")
        
    return res.fun

def wassersteinrDistGMM(gmm1,gmm2):
    n1 = gmm1.Ncomp
    n2 = gmm2.Ncomp
    D=np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            D[i,j] = WassersteinDist_gaussian(gmm1.getCompPDF(i,asScipy=False),gmm2.getCompPDF(j,asScipy=False))
            
            
    A=[]
    for i in range(n1):
        z = np.zeros(n1*n2)
        z[i*n2:(i+1)*n2] = np.ones(n2)
        A.append(z)
    
    B=[]
    for i in range(n1):
        B.append(np.identity(n2))
    
    Aeq = np.vstack([A,np.hstack(B)])    
    Beq = np.hstack([gmm1.wts,gmm2.wts])
    
    # print(Aeq)
    # print(Beq)
    # print(D)
    warnings.filterwarnings("ignore")
    DD = D.reshape(-1)
    bnds=[]
    for i in range(len(DD)):
        bnds.append((0,1))    
    res = scopt.linprog(DD, A_eq=Aeq, b_eq=Beq)
    # print(res)
    warnings.filterwarnings("default")
        
    return res.fun

def hellingerDistPoints(X,w1,w2,D=None):
    n1 = X.shape[0]
    n2 = X.shape[0]
    if D is None:
        D=np.zeros((n1,n2))
        for i in range(n1):
            D[i] = nplg.norm(X-X[i],axis=1)
            
            
    A=[]
    for i in range(n1):
        z = np.zeros(n1*n2)
        z[i*n2:(i+1)*n2] = np.ones(n2)
        A.append(z)
    
    B=[]
    for i in range(n1):
        B.append(np.identity(n2))
    
    Aeq = np.vstack([A,np.hstack(B)])    
    Beq = np.hstack([w1,w2])
    
    # print(Aeq)
    # print(Beq)
    # print(D)
    warnings.filterwarnings("ignore")
    res = scopt.linprog(D.reshape(-1), A_ub=None, b_ub=None, A_eq=Aeq, b_eq=Beq)
    # print(res)
    warnings.filterwarnings("default")
        
    return res.fun

    # np.hstack([gmm1.wts- sum(W,axis=1), gmm2.wts- sum(W,axis=0)])

def mutualInformation_covs(P1,P2):
    """
    Generate the mutual inforation 0.5 log(det(P1)/det(P2))

    Parameters
    ----------
    P1 : TYPE
        DESCRIPTION.
    P2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    egval1,egvec1 = nplg.eig(P1)
    egval2,egvec2 = nplg.eig(P2)
    egval1log = np.log( np.sort(egval1) )
    egval2log = np.log( np.sort(egval2) )
    MI=0.5*( np.sum(egval1log) - np.sum(egval2log) )
    
    return MI

# if __name__ == "__main__":
#     import matplotlib
#     try:
#         matplotlib.use('TkAgg')
#     except:
#         matplotlib.use('Qt5Agg')
#     import matplotlib.pyplot as plt
#     from utils.plotting import surface as utlsrf
#     from utils.plotting import geometryshapes as utgeosh
    
#     m1 = np.array([0,0])
#     P1 = np.array([[5,-2],[-2,5]])
    
#     m2 = np.array([0,0])
#     P2 = np.array([[1,-0.5],[-0.5,1]])
    
#     g1 = multivariate_normal(m1,P1)
#     g2 = multivariate_normal(m2,P2)
    
#     xx1,yy1,p1 = utlsrf.plotpdf2Dsurf(g1,Ng=50)
#     xx2,yy2,p2 = utlsrf.plotpdf2Dsurf(g2,Ng=50)
    
#     c1 = utgeosh.getCovEllipsePoints2D(m1,P1,nsig=1,N=100)
#     c2 = utgeosh.getCovEllipsePoints2D(m2,P2,nsig=1,N=100)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     ax.plot(c1[:,0],c1[:,1],'r')
#     ax.plot(c2[:,0],c2[:,1],'b')
    

#     ax.plot_surface(xx1,yy1,p1,alpha=0.8,linewidth=1)
#     ax.plot_surface(xx2,yy2,p2,alpha=0.8,linewidth=1)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     print(hellingerDist(g1,g2))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    