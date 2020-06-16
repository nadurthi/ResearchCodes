
import numpy as np
import numpy.linalg as nplnalg
import scipy.linalg as sclnalg

def getEllipsePoints2D(xyc,a,b,N=100):
    th = np.linspace(0,2*np.pi,N)
    X = np.zeros((N,2))
    for i in range(len(th)):    
        X[i,:]=xyc+[a*np.cos(th[i]),b*np.sin(th[i])];
    
    return X



def getCirclePoints2D(xyc,r,N=100):
    X = getEllipsePoints2D(xyc,r,r,N=N)
    return X


def getCovEllipsePoints2D(m,P,nsig=1,N=100):
    mz = np.zeros(m.shape)
    A = sclnalg.sqrtm(P) 
    X = getCirclePoints2D(mz,nsig,N=N)
    
    for i in range(X.shape[0]):    
        X[i,:]=m+np.matmul(A,X[i,:])
    
    return X