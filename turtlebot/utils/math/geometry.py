# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nplg
import scipy.linalg as sclnalg
from numpy.linalg import multi_dot

def isinside_points_circle(X,xc,r):
    if X.ndim==1:
        y = nplg.norm(X-xc)<r
    else:
        y = nplg.norm(X-xc,axis=1)<r
    return y #.astype(int)

def isinside_points_nsigcov(X,m,P,n):
    N=X.shape[0]
    A = nplg.inv(n**2*P)
    y = np.zeros(N)
    for i in range(N):
        y[i] = np.matmul(np.matmul(X[i,:]-m,A),X[i,:]-m)
    return y<=1



def isoverlap_ellipse_ellipse(xc1,a1,b1,xc2,a2,b2,minfrac=5):
    """
    Check if two ellipses intersect
    discretize one and see if the points fall in another
    Parameters
    ----------
    xc1 : TYPE
        DESCRIPTION.
    a1 : TYPE
        DESCRIPTION.
    b1 : TYPE
        DESCRIPTION.
    xc2 : TYPE
        DESCRIPTION.
    a2 : TYPE
        DESCRIPTION.
    b2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    N = 1000
    th = np.linspace(0,2*np.pi,N)
    # X = np.zeros((N,2))
    X1 = np.vstack([a1*np.cos(th),b1*np.sin(th)]).T
    X1[:,0] = X1[:,0] + xc1[0]
    X1[:,1] = X1[:,1] + xc1[1]
    
    X2 = np.vstack([a2*np.cos(th),b2*np.sin(th)]).T
    X2[:,0] = X2[:,0] + xc2[0]
    X2[:,1] = X2[:,1] + xc2[1]
    
    y1in2=np.power((X1[:,0]-xc2[0]),2)/a2**2+np.power((X1[:,1]-xc2[1]),2)/b2**2-1
    y2in1=np.power((X2[:,0]-xc1[0]),2)/a1**2+np.power((X2[:,1]-xc1[1]),2)/b1**2-1
    
    return 100*sum(y1in2<0)/N > minfrac or 100*sum(y2in1<0)/N > minfrac
    
    
def isoverlap_circle_ellipse(xcircle,r,xc2,a2,b2,minfrac=5):
    return isoverlap_ellipse_ellipse(xcircle,r,r,xc2,a2,b2,minfrac=minfrac)


# def isoverlap_0circle_cov(n1,m,P,n2):
#     N = 1000
#     th = np.linspace(0,2*np.pi,N)
#     # X = np.zeros((N,2))
#     Xcircle = np.vstack([n1*np.cos(th),n1*np.sin(th)]).T
#     A = nplg.inv(n2**2*P)
#     y = np.zeros(N)
#     for i in range(N):
#         y[i] = np.matmul(np.matmul(Xcircle[i,:]-m,A),Xcircle[i,:]-m)

#     return any(y<=1)

def isoverlap_circle_cov(xc,r,m,P,n,minfrac=5):
    N = 1000
    th = np.linspace(0,2*np.pi,N)
    # X = np.zeros((N,2))
    Xcircle = np.vstack([r*np.cos(th)+xc[0],r*np.sin(th)+xc[1]]).T
    
    A = sclnalg.sqrtm(n**2*P)
    Ainv = nplg.inv(n**2*P)
    
    Xsigcov = np.vstack([np.cos(th),np.sin(th)]).T
    for i in range(N):
        Xsigcov[i,:] = np.matmul(A,Xsigcov[i,:])+m    
    
    # y = np.zeros(N)
    # for i in range(N):
    #     y[i] = np.matmul(np.matmul(Xcircle[i,:]-m,A),Xcircle[i,:]-m)
    # E is ellipse of n-sig, C is cirle
    yEinC=np.power((Xsigcov[:,0]-xc[0]),2)+np.power((Xsigcov[:,1]-xc[1]),2)-r**2
    yCinE = np.zeros(N)
    for i in range(N):
        yCinE[i]=np.matmul(np.matmul(Xcircle[i]-m,Ainv),Xcircle[i]-m)
    
    return 100*sum(yEinC<0)/N > minfrac or 100*sum(yCinE<1)/N > minfrac



def isoverlap_cov_cov(m1,P1,n1,m2,P2,n2,minfrac=5):
    """
    ckeck if the n1 sigma ellipsoid intersects with the n2 sigma ellipsoid of second covariance

    Parameters
    ----------
    m1 : TYPE
        DESCRIPTION.
    P1 : TYPE
        DESCRIPTION.
    n1 : TYPE
        DESCRIPTION.
    m2 : TYPE
        DESCRIPTION.
    P2 : TYPE
        DESCRIPTION.
    n2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    A = nplg.inv(sclnalg.sqrtm(P1))
    
    # m1n = A*(m1-m1)
    # P1n = multi_dot([A,P1,A.T])
    
    m2n = A*(m2-m1)
    P2n = multi_dot([A,P2,A.T])
    
    return isoverlap_circle_cov(0*m1,n1,m2n,P2n,n2,minfrac=minfrac)



def isIntersect_circle_cov(xc,r,m,P,n2):
    N = 1000
    th = np.linspace(0,2*np.pi,N)
    # X = np.zeros((N,2))
    Xcircle = np.vstack([r*np.cos(th)+xc[0],r*np.sin(th)+xc[1]]).T
    A = nplg.inv(n2**2*P)
    y = np.zeros(N)
    for i in range(N):
        y[i] = np.matmul(np.matmul(Xcircle[i,:]-m,A),Xcircle[i,:]-m)

    return not (np.all(y<1) or np.all(y>1))






def isIntersect_cov_cov(m1,P1,n1,m2,P2,n2):
    N = 1000
    th = np.linspace(0,2*np.pi,N)
    # X = np.zeros((N,2))
    Xcircle = np.vstack([np.cos(th)+xc[0],np.sin(th)+xc[1]]).T

    A1 = sclnalg.sqrtm(n1**2*P1)
    A2inv = nplg.inv(n2**2*P2)
    
    X1 = np.zeros((N,2))
    for i in range(N):
        X1[i,:] = np.matmul(A1,Xcircle[i,:])+m1   

        
    y1in2 = np.zeros(N)
    for i in range(N):
        y1in2[i] = np.matmul(np.matmul(X1[i,:]-m2,A2inv),X1[i,:]-m2)

        
    return not (np.all(y1in2<1) or np.all(y1in2>1))


if __name__ == "__main__":
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    
    
    xc1=np.array([1,1])
    xc2=np.array([1,-1])
    a1 = 2
    b1 = 1
    a2 = 2
    b2 = 0.9
    
    print("Ellipse to Ellipse overlap")
    print( isoverlap_ellipse_ellipse(xc1,a1,b1,xc2,a2,b2,minfrac=5))
    
    
    
    xc=np.array([1,1])
    r=1
    m=np.array([3,3])
    P=np.array([[1,0.5],[0.5,1]])
    
    
    
    
    
    