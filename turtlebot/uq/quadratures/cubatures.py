#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""


import os
import scipy.io as sio
import scipy.linalg as sclg
import numpy.linalg as nplg
import logging
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


dirname = os.path.dirname(__file__)
dataGaussianFolder = os.path.join(dirname, 'data', 'gaussian_mat')
dataUniformFolder = os.path.join(dirname, 'data', 'uniform_mat')

from enum import Enum, auto

class SigmaMethod(Enum):
    UT = auto()
    MC = auto()
    CUT4 = auto()
    CUT6 = auto()
    CUT8 = auto()
    GH2 = auto()
    GH3 = auto()
    GH4 = auto()
    GH5 = auto()
    GH6 = auto()
    GH7 = auto()



# %%
# Cache to remember the points

@lru_cache(maxsize=None)
def getGaussianPointFromFile(method,file):
    
    mat_fname = os.path.join(dataGaussianFolder, method, file)
    mat_contents = sio.loadmat(mat_fname)
    return mat_contents

# %%
def chebychevextrema(m):
    """
    Generate chebychev points
    """
    X = np.zeros(m, 1)
    if m == 1:
        X = 0
        return X

    X = np.zeros(m)
    for j in range(m):
        X[j] = -np.cos(np.pi * (j - 1) / (m - 1))

    X = np.round(X * 1e14) / 1e14
    return X


# %%
def UT_sigmapoints(mu, P):

    kappa = 1
    n = len(mu)
    # % m=2 is 2n+1 points
    if n < 4:
        k = 3 - n
    else:
        k = 1

    x = np.zeros((2 * n + 1, n))
    w = np.zeros(2 * n + 1)
    x[0, :] = mu
    w[0] = k / (n + k)

    A = sclg.sqrtm((n + k) * P)

    for i in range(n):
        x[i + 1, :] = (mu + A[:, i])
        x[i + n + 1, :] = (mu - A[:, i])
        w[i + 1] = 1 / (2 * (n + k))
        w[i + n + 1] = 1 / (2 * (n + k))

    return (x, w)

def UT2n_sigmapoints(mu, P):


    n = len(mu)
    k=0

    x = np.zeros((2 * n, n))
    w = np.zeros(2 * n )

    A = sclg.sqrtm((n + k) * P)

    for i in range(n):
        x[i , :] = (mu + A[:, i])
        x[i + n , :] = (mu - A[:, i])
        w[i ] = 1 / (2 * (n + k))
        w[i + n ] = 1 / (2 * (n + k))

    return (x, w)
# %%

def CUT4pts_gaussian(mu, P):
    """
    generate points for CUT4
    """
    n = len(mu)
    file = 'cut4_%dD_gaussian.mat' % n
    mat_contents = getGaussianPointFromFile('CUT',file)
    X=np.zeros(mat_contents['X'].shape)
    A = sclg.sqrtm(P)
    for i in range(X.shape[0]):
        X[i,:] = np.matmul(A,mat_contents['X'][i,:]) + mu
    
    return (X, mat_contents['w'].reshape(-1).copy())


def CUT6pts_gaussian(mu, P):
    n = len(mu)
    file = 'cut6_%dD_gaussian.mat' % n
    mat_contents = getGaussianPointFromFile('CUT',file)
    X=np.zeros(mat_contents['X'].shape)
    A = sclg.sqrtm(P)
    for i in range(X.shape[0]):
        X[i,:] = np.matmul(A,mat_contents['X'][i,:]) + mu
        
    return (X, mat_contents['w'].reshape(-1))


def CUT8pts_gaussian(mu, P):
    n = len(mu)
    file = 'cut8_%dD_gaussian.mat' % n
    mat_contents = getGaussianPointFromFile('CUT',file)
    X=np.zeros(mat_contents['X'].shape)
    A = sclg.sqrtm(P)
    for i in range(X.shape[0]):
        X[i,:] = np.matmul(A,mat_contents['X'][i,:]) + mu
        
    return (X, mat_contents['w'].reshape(-1))


def GH_points(mu, P, N):
    n = len(mu)
    file = 'GH%d_%dD_gaussian.mat' % (N,n)
    mat_contents = getGaussianPointFromFile('GH',file)
    X=np.zeros(mat_contents['X'].shape)
    A = sclg.sqrtm(P)
    for i in range(X.shape[0]):
        X[i,:] = np.matmul(A,mat_contents['X'][i,:]) + mu
        
    return (X, mat_contents['w'].reshape(-1))


GaussianSigmaPtsMethodsDict = {
    SigmaMethod.UT: UT_sigmapoints,
    SigmaMethod.CUT4: CUT4pts_gaussian,
    SigmaMethod.CUT6: CUT6pts_gaussian,
    SigmaMethod.CUT8: CUT8pts_gaussian
    }
# %%

def GH1Dpoints(m,var,N):
    if N==2:
        x=[-1.0000, 1.0000]
        w=[0.5000, 0.5000]
    if N==3:
        x=[-1.7321,0,1.7321]
        w=[0.1667,0.6667,0.1667]
    if N==4:
        x=[-2.3344  , -0.7420  ,  0.7420 ,   2.3344]
        w=[0.0459  ,  0.4541   , 0.4541  ,  0.0459]
    if N==5:
        x=[-2.8570,   -1.3556,         0,    1.3556,    2.8570]
        w=[0.0113,    0.2221 ,   0.5333 ,   0.2221 ,   0.0113]
    if N==6:
        x=[-3.3243 ,  -1.8892 ,  -0.6167,    0.6167 ,   1.8892,    3.3243]
        w=[0.0026   , 0.0886   , 0.4088  ,  0.4088  ,  0.0886 ,   0.0026]
    if N==7:
        x=[-3.7504 ,  -2.3668 ,  -1.1544,         0  ,  1.1544 ,   2.3668  ,  3.7504]
        w=[0.0005  ,  0.0308  ,  0.2401 ,   0.4571  ,  0.2401  ,  0.0308   , 0.0005]
        
    sig = np.sqrt(var)
    x = sig*np.array(x)+m
    w = np.array(w)
    return x,w 

# %%
# def smolyak_sparse_grid_modf(mu,P,d,l,type):
# # ======================================================================
# #     % d is the dimension of the system
# #     % type is eith 'GH' for gauss hermite
# #     %              'GLgn'for gauss Legendre
# #     % l is the number of points of the method type in 1 dimension.
# #     % therefore the smolyak sparse grid scheme produces a
# #     % quadrature rule that can integrate all polynomials of total degree
# #     % 2l-1.
# # ======================================================================

#     switch lower(type)
#         case 'gh'
#             QUADD=@(m)GH_points(0,1,m);
#         case 'glgn'
#             QUADD=@(m)GLeg_pts(m, -1, 1);
#         case 'patglgn'
#             QUADD=@(m)patterson_rule ( 2*m-1, -1, 1 );
#         otherwise
#             disp('Gods must be crazy; They made you!!!')
#             return;
#     end

#     if d==1
#         [x,w]=QUADD(l);
#         return
#     end

#     x=[];
#     w=[];
#     for i=1:1:l
#         [x1,w1]=QUADD(i);
#         if i-1>0
#         [x2,w2]=QUADD(i-1);
#         x2=-x2;
#         w2=-w2;
#         [xd1,wd1]=smolyak_sparse_grid(d-1,l-i+1,type);
#         [xd,wd]=tens_prod_vec(x1,xd1,w1,wd1);
#         [xdm,wdm]=tens_prod_vec(x2,xd1,w2,wd1);
#         x=vertcat(x,xd);
#         w=vertcat(w,wd);
#         x=vertcat(x,xdm);
#         w=vertcat(w,wdm);

#         else
#          [xd1,wd1]=smolyak_sparse_grid(d-1,l-i+1,type);
#          [xd,wd]=tens_prod_vec(x1,xd1,w1,wd1);
#          x=vertcat(x,xd);
#          w=vertcat(w,wd);
#         end

#     end

#     %% Now finding the duplicate points and adding their weights
#     ep=1e-12;
#     i=1;
#     while i<=length(w)
#     %     if i>length(w)
#     %         break
#     %     end
#         I=find(sum((x-repmat(x(i,:),size(x,1),1)).^2,2)<ep);
#         if length(I)>1
#         w(i)=sum(w(I));
#         x(I(2:end),:)=[];
#         w(I(2:end))=[];
#         end
#     i=i+1;
#     end

#     %   w=w/sum(w);

#     A=sqrtm(P);
#     for i=1:1:length(w)
#         x(i,:)=A*x(i,:)'+mu(:);
#     end

#     return x,w


# # %%
# def GLgn_pts(bdd_low,bdd_up,Np):
#     n=length(bdd_low);

#     if Np==1
#         x=1;
#         w=1;
#     end

#     if Np==2
#         x=(roots([3,0,-1]));
#         B=[1,0,1/3]';
#         A=[1,1;
#           x(1),x(2);
#           x(1)^2,x(2)^2];
#         w=A\B;
#     end

#     if Np==3
#        x=roots([5,0,-3,0]);
#            B=[1,0,1/3,0,1/5]';
#         A=[1,1,1;
#           x(1),x(2),x(3);
#           x(1)^2,x(2)^2,x(3)^2;
#           x(1)^3,x(2)^3,x(3)^3;
#           x(1)^4,x(2)^4,x(3)^4];
#         w=A\B;
#     end

#     if Np==4
#        x=roots([35,0,-30,0,3]);
#               B=[1,0,1/3,0,1/5,0,1/7]';
#         A=[1,1,1,1;
#           x(1),x(2),x(3),x(4);
#           x(1)^2,x(2)^2,x(3)^2,x(4)^2;
#           x(1)^3,x(2)^3,x(3)^3,x(4)^3;
#           x(1)^4,x(2)^4,x(3)^4,x(4)^4;
#           x(1)^5,x(2)^5,x(3)^5,x(4)^5;
#           x(1)^6,x(2)^6,x(3)^6,x(4)^6];
#         w=A\B;
#     end

#     if Np==5
#        x=roots([63,0,-70,0,15,0]);
#                  B=[1,0,1/3,0,1/5,0,1/7,0,1/9]';
#         A=[1,1,1,1,1;
#           x(1),x(2),x(3),x(4),x(5);
#           x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2;
#           x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3;
#           x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4;
#           x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5;
#           x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6;
#           x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7;
#           x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8];
#         w=A\B;
#     end

#     if Np==6
#        x=roots([231,0,-315,0,105,0,-5]);
#                  B=[1,0,1/3,0,1/5,0,1/7,0,1/9,0,1/11]';
#         A=[1,1,1,1,1,1;
#           x(1),x(2),x(3),x(4),x(5),x(6);
#           x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2;
#           x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3;
#           x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4;
#           x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5;
#           x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6;
#           x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7;
#           x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8;
#           x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9;
#           x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10];
#         w=A\B;
#     end

#     if Np==7
#        x=roots([429,0,-693,0,315,0,-35,0]);
#                     B=[1,0,1/3,0,1/5,0,1/7,0,1/9,0,1/11,0,1/13]';
#         A=[1,1,1,1,1,1,1;
#           x(1),x(2),x(3),x(4),x(5),x(6),x(7);
#           x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2,x(7)^2;
#           x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3,x(7)^3;
#           x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4,x(7)^4;
#           x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5,x(7)^5;
#           x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6,x(7)^6;
#           x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7,x(7)^7;
#           x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8,x(7)^8;
#           x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9,x(7)^9;
#           x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10,x(7)^10;
#           x(1)^11,x(2)^11,x(3)^11,x(4)^11,x(5)^11,x(6)^11,x(7)^11;
#           x(1)^12,x(2)^12,x(3)^12,x(4)^12,x(5)^12,x(6)^12,x(7)^12];
#         w=A\B;
#     end

#     if Np==8
#        x=roots([6435,0,-12012,0,6930,0,-1260,0,35]);
#                        B=[1,0,1/3,0,1/5,0,1/7,0,1/9,0,1/11,0,1/13,0,1/15]';
#         A=[1,1,1,1,1,1,1,1;
#           x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8);
#           x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2,x(7)^2,x(8)^2;
#           x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3,x(7)^3,x(8)^3;
#           x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4,x(7)^4,x(8)^4;
#           x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5,x(7)^5,x(8)^5;
#           x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6,x(7)^6,x(8)^6;
#           x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7,x(7)^7,x(8)^7;
#           x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8,x(7)^8,x(8)^8;
#           x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9,x(7)^9,x(8)^9;
#           x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10,x(7)^10,x(8)^10;
#           x(1)^11,x(2)^11,x(3)^11,x(4)^11,x(5)^11,x(6)^11,x(7)^11,x(8)^11;
#           x(1)^12,x(2)^12,x(3)^12,x(4)^12,x(5)^12,x(6)^12,x(7)^12,x(8)^12;
#           x(1)^13,x(2)^13,x(3)^13,x(4)^13,x(5)^13,x(6)^13,x(7)^13,x(8)^13;
#           x(1)^14,x(2)^14,x(3)^14,x(4)^14,x(5)^14,x(6)^14,x(7)^14,x(8)^14];
#         w=A\B;
#     end

#     if Np==9
#        x=roots([12155,0,-25740,0,18018,0,-4620,0,315,0]);
#                           B=[1,0,1/3,0,1/5,0,1/7,0,1/9,0,1/11,0,1/13,0,1/15,0,1/17]';
#         A=[1,1,1,1,1,1,1,1,1;
#           x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9);
#           x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2,x(7)^2,x(8)^2,x(9)^2;
#           x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3,x(7)^3,x(8)^3,x(9)^3;
#           x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4,x(7)^4,x(8)^4,x(9)^4;
#           x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5,x(7)^5,x(8)^5,x(9)^5;
#           x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6,x(7)^6,x(8)^6,x(9)^6;
#           x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7,x(7)^7,x(8)^7,x(9)^7;
#           x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8,x(7)^8,x(8)^8,x(9)^8;
#           x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9,x(7)^9,x(8)^9,x(9)^9;
#           x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10,x(7)^10,x(8)^10,x(9)^10;
#           x(1)^11,x(2)^11,x(3)^11,x(4)^11,x(5)^11,x(6)^11,x(7)^11,x(8)^11,x(9)^11;
#           x(1)^12,x(2)^12,x(3)^12,x(4)^12,x(5)^12,x(6)^12,x(7)^12,x(8)^12,x(9)^12;
#           x(1)^13,x(2)^13,x(3)^13,x(4)^13,x(5)^13,x(6)^13,x(7)^13,x(8)^13,x(9)^13;
#           x(1)^14,x(2)^14,x(3)^14,x(4)^14,x(5)^14,x(6)^14,x(7)^14,x(8)^14,x(9)^14;
#           x(1)^15,x(2)^15,x(3)^15,x(4)^15,x(5)^15,x(6)^15,x(7)^15,x(8)^15,x(9)^15;
#           x(1)^16,x(2)^16,x(3)^16,x(4)^16,x(5)^16,x(6)^16,x(7)^16,x(8)^16,x(9)^16];
#         w=A\B;
#     end

#     if Np==10
#        x=roots([46189,0,-109395,0,90090,0,-30030,0,3465,0,-63]);
#                           B=[1,0,1/3,0,1/5,0,1/7,0,1/9,0,1/11,0,1/13,0,1/15,0,1/17,0,1/19]';
#         A=[1,1,1,1,1,1,1,1,1,1;
#           x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10);
#           x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2,x(7)^2,x(8)^2,x(9)^2,x(10)^2;
#           x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3,x(7)^3,x(8)^3,x(9)^3,x(10)^3;
#           x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4,x(7)^4,x(8)^4,x(9)^4,x(10)^4;
#           x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5,x(7)^5,x(8)^5,x(9)^5,x(10)^5;
#           x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6,x(7)^6,x(8)^6,x(9)^6,x(10)^6;
#           x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7,x(7)^7,x(8)^7,x(9)^7,x(10)^7;
#           x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8,x(7)^8,x(8)^8,x(9)^8,x(10)^8;
#           x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9,x(7)^9,x(8)^9,x(9)^9,x(10)^9;
#           x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10,x(7)^10,x(8)^10,x(9)^10,x(10)^10;
#           x(1)^11,x(2)^11,x(3)^11,x(4)^11,x(5)^11,x(6)^11,x(7)^11,x(8)^11,x(9)^11,x(10)^11;
#           x(1)^12,x(2)^12,x(3)^12,x(4)^12,x(5)^12,x(6)^12,x(7)^12,x(8)^12,x(9)^12,x(10)^12;
#           x(1)^13,x(2)^13,x(3)^13,x(4)^13,x(5)^13,x(6)^13,x(7)^13,x(8)^13,x(9)^13,x(10)^13;
#           x(1)^14,x(2)^14,x(3)^14,x(4)^14,x(5)^14,x(6)^14,x(7)^14,x(8)^14,x(9)^14,x(10)^14;
#           x(1)^15,x(2)^15,x(3)^15,x(4)^15,x(5)^15,x(6)^15,x(7)^15,x(8)^15,x(9)^15,x(10)^15;
#           x(1)^16,x(2)^16,x(3)^16,x(4)^16,x(5)^16,x(6)^16,x(7)^16,x(8)^16,x(9)^16,x(10)^16;
#           x(1)^17,x(2)^17,x(3)^17,x(4)^17,x(5)^17,x(6)^17,x(7)^17,x(8)^17,x(9)^17,x(10)^17;
#           x(1)^18,x(2)^18,x(3)^18,x(4)^18,x(5)^18,x(6)^18,x(7)^18,x(8)^18,x(9)^18,x(10)^18];
#         w=A\B;
#     end


#     %% take tensor product of the points and weights
#     X=x;
#     W=w;
#     for i=1:1:n-1
#     [X,W]=tens_prod_vec(X,x,W,w);
#     end

#     %% transforming the points
#     mu=(bdd_low+bdd_up)/2;
#     h=-bdd_low+bdd_up;
#     for i=1:1:n
#         X(:,i)=(h(i)/2)*X(:,i)+mu(i);
#     end


# %%
