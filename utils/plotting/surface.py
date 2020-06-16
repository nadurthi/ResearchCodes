# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as nplg
import scipy.linalg as sclnalg

def plotpdf2Dsurf(pdf,Ng=50):
    m = pdf.mean
    P = pdf.cov
    
    A = sclnalg.sqrtm(P) 
    xg = np.linspace(m[0]-4*A[0,0],m[0]+4*A[0,0],Ng)
    yg = np.linspace(m[1]-4*A[1,1],m[1]+4*A[1,1],Ng)
    xx,yy = np.meshgrid(xg,yg )
    X=np.hstack([xx.reshape(-1,1),yy.reshape(-1,1)])
    p = pdf.pdf(X)
    p = p.reshape(Ng,Ng)
    
    return xx,yy,p