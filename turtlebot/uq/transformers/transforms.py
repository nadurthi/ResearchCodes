#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

# import numpy.matlib as npmt
import numpy as np
import numpy.linalg as nplg
import scipy.linalg as sclg
import uuid
import logging
from uq.stats import moments as uqstat
import copy
import ipdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class XProbdataSet:
    def __init__(self):
        self.ID = uuid.uuid4()
        self.stagesOrig2Final = []
        
    def fit_apply_normalize_0I(self,X,p,mu=None,cov=None):
        if mu is None or cov is None:
            mu,cov = uqstat.MeanCov(X, p/np.sum(p))
        # ipdb.set_trace()
        
        A = nplg.inv(sclg.sqrtm(cov))
        m = -np.matmul(A,mu)
        XX = affineTransform(X,A,m)
        
        pp =  p/nplg.det(A)
        
        self.stagesOrig2Final.append({'func':'affineTransform','mu':mu,'cov':cov,'A':A,'m':m})  
        
        return XX,pp
    
    def applyForward(self,X,p):
        for d in self.stagesOrig2Final:
            if d['func'] == 'affineTransform':
                XX = affineTransform(X,d['A'],d['m'])
                pp =  p/nplg.det(d['A'])
        return XX,pp
    
    def applyBackward(self,X,p):
        for d in self.stagesOrig2Final[::-1]:
            if d['func'] == 'affineTransform':
                A = nplg.inv(d['A'])
                m = -np.matmul(A,d['m'])
                XX = affineTransform(X,A,m)
                pp =  p/nplg.det(A)
        return XX,pp
        
        
    def makeCopy(self):
        xpds =copy.deepcopy(self)
        
        
        return xpds
    
    
    
    
def affineTransform(X, A, MU):
    if X.ndim == 1:
        X = X.reshape((1, X.size))

    n, dim = X.shape

    Y = np.zeros(X.shape)
    for i in range(n):
        Y[i, :] = np.matmul(A, X[i, :]) + MU

    if n == 1:
        Y = Y.reshape(-1)

    return Y


def transform_rect_domain(Xb,bl,bu,Bl,Bu):

    ns,nx=Xb.shape
       
    rb=0.5*(bl+bu);
    # %shifting to 0 mean
    X0=Xb-rb
    
    # %scaling the points
    db = bu-bl
    dB = Bu-Bl
    
    d = dB/db
    
    Xs = d*X0
    
    # %shifting the mean
    
    rB=0.5*(Bl+Bu);
    # % rmu=rB-rb;
    
    XB=Xs+rB
    
    return XB,d



# %%

#
#function[nodes, weights] = tensor_product(n1D, w1D)
#
#%*****************************************************************************80
#%
#% % TENSOR_PRODUCT generates a tensor product quadrature rule.
#%
#%  Discussion:
#%
#%    The Kronecker product of an K by L matrix A and an M by N matrix B
#% is the K * M by L * N matrix formed by
#%
#%      a(1, 1) * B, a(1, 2) * B, ..., a(1, l) * B
#%      a(2, 1) * B, a(2, 2) * B, ..., a(2, l) * B
#%      ..........   ..........   .... ..........
#%      a(k, 1) * B, a(k, 2) * B, ..., a(k, l) * B
#%
#%    Thanks to Ivan Ukhov for pointing out a tiny but deadly typographical
#%    error, 17 July 2012.
#%
#%  Licensing:
#%
#%    This code is distributed under the GNU LGPL license.
#%
#%  Modified:
#%
#%    17 July 2012
#%
#%  Author:
#%
#%    Original MATLAB version by Florian Heiss, Viktor Winschel.
#%    This MATLAB version by John Burkardt.
#%
#%  Reference:
#%
#%    Florian Heiss, Viktor Winschel,
#%    Likelihood approximation by numerical integration on sparse grids,
#%    Journal of Econometrics,
#%    Volume 144, 2008, pages 62 - 80.
#%
#%  Parameters:
#%
#%    Input, cell array n1D{}, contains K sets of 1D nodes.
#%    The I - th set has dimension N(I).
#%    Each entry of the cell array should be a column vector.
#%
#%    Input, cell array w1D{}, contains K sets of 1D weights.
#%
#%    Output, real nodes(NPROD, K), the tensor product nodes.
#%    NPROD = product(N(1) * ... * N(K)).
#%
#%    Output, real weights(NPROD), the tensor product weights.
#%
#  dimension = length(n1D);
#
#  nodes = n1D{1};
#  nodes = nodes (:);
#  weights = w1D{1};
#  weights = weights (:);
#
#  for j = 2: dimension
#
#    newnodes = n1D{j};
#    newnodes = newnodes (:);
#
#    a = ones(size(newnodes, 1), 1);
#    b = ones(size(nodes, 1), 1);
#    c = kron(nodes, a);
#    d = kron(b, newnodes);
#
#    nodes = [c, d];
#
#    newweights = w1D{j};
#    newweights = newweights (:);
#    weights = kron(weights, newweights);
#
#  end
#
#  return
#end
#
#
## %%
#
#
#function[T, W] = tens_prod_vec(u, v, wu, wv)
#% 1 enitity is one row of any matrix u or v
#% the rows of u are tensors producted with rows of v
#
#if isempty(u)
#    T = v;
#    W = wv;
#    return
#end
#if isempty(v)
#    T = u;
#    W = wu;
#    return
#end
#
#n = size(u, 1);
#m = size(v, 1);
#T = [];
#W = [];
#for i = 1: 1: n
#    T = vertcat(T, horzcat(repmat(u(i, : ), m, 1), v));
#    W = vertcat(W, horzcat(repmat(wu(i), m, 1), wv));
#end
#W = prod(W, 2);
#% W = W / sum(W);
#% uc = size(u, 2);
#% vc = size(v, 2);
#% T = zeros(n * m, size(u, 2) + size(v, 2));
#% k = 1;
#% for i = 1: 1: n
#%     T(k, 1: uc) = u(i, : );
#% for j = 1: 1: m
#%         T(k, uc+1: uc+vc) = v(j, : );
#%     end
#%     k = k + 1;
#% end
#
#
## %%
#
#function[mu, P] = cal_moms_wrt_allstates(X, w, N, type)
#% given X(time, states, samples) and constant w - weight calculate the moments
#% with respect to time for all states
#% N is the the order of required moment
#nt = size(X, 1); % no. of time steps
#nx = size(X, 2); % no. of states
#ns = size(X, 3); % no. of samples
#if size(w, 1) == 1
#    w = w';
#end
#% calculating the mean
#
#% switch between central and mean by substracting the mean from the samples
#% if strcmp(type, 'central') == 1
#%    X = X - mu;
#% end
#P = zeros(nt, nx ^ 2);
#mu = zeros(nt, nx);
#for i = 1: 1: nt
#    for j = 1: 1: ns
#        x(j, :) = X(i, : , j);
#    end
#    W = repmat(w, 1, nx);
#    mk = sum(W.*x, 1)';
#    MU = repmat(mk', ns, 1);
#    xx = x - MU;
#    Pk = xx'* (W.*xx);
#    mu(i, : ) = mk;
#    P(i, : ) = reshape(Pk, 1, nx^2);
#end
#end
#
#
## %%
#function m = cal_moments_wrt_pts(x, w, NN)
#%Calculate all the moments till order NN
#[mmm, mm] = size(x);
#%Dimension of the system
#n = mm;
#m = [];
#for N = 2: 2: NN
#
#    combos = GenerateIndex(n, (N + 1) * ones(1, n));
#combos(find(combos == (N + 1))) = 0;
#
#y = [];
#for i = 1: 1: length(combos)
#    if sum(combos(i, : )) == N
#     y = vertcat(y, combos(i, :));
#%      x = vertcat(x, wrev(combos(i, :)));
#    end
#end
#y = sortrows(y, -1);
#[yy, yyy] = size(y);
#
#c = [];
#
#for i = 1: 1: yy
#    p = w;
#    for j = 1: 1: n
#        p = p.*x(:, j).^y(i, j);
#    end
#    p = sum(p);
#    c = vertcat(c, p);
#end
#% if N == 2
#%     m = vertcat(m, [y, c, ones(length(c), 1)]);
#% else
#% g = permute_moments(P, N);
#
#m = vertcat(m, [y, c]);
#% end
#end
#
#
## %%
#
#function[y, M] = Cal_moments_samples(X, w, N, type)
#% given X(samples, states) and constant w - weight calculate the moments
#
#% N is the the order of required moment
#nx = size(X, 2); % no. of states
#ns = size(X, 1); % no. of samples
#if size(w, 1) == 1
#    w = w';
#end
#W = repmat(w, 1, nx);
#mu = sum(W.*X, 1);
#if N == 1
#    M = mu';
#    y = eye(nx);
#    return;
#end
#if strcmp(type, 'central') == 1
#  disp('central moms')
#    X = X - repmat(mu, ns, 1);
#
#elseif strcmp(type, 'raw') == 1
#     disp('raw moms')
#else
#    disp('err')
#    return;
#end
#% % % % % % % % % % % % % % % % % % % % % % % % % % % % %%
#combos = GenerateIndex(nx, (N + 1) * ones(1, nx));
#combos(find(combos == (N + 1))) = 0;
#
#y = [];
#for i = 1: 1: length(combos)
#    if sum(combos(i, : )) == N
#     y = vertcat(y, combos(i, :));
#%      x = vertcat(x, wrev(combos(i, :)));
#    end
#end
#y = sortrows(y, -1);
#
#[yy, yyy] = size(y);
#
#% % % % % % % % % % % % % % % % % % % % % % % % % % %%
#
#M = zeros(yy, 1);
#
#        for i = 1: 1: yy
#        M(i) = sum(w.*prod(X.^repmat(y(i, :), ns, 1), 2));
#        end
#
#
## %%
#
#
#function m = cal_all_moms(X, w, N, type)
#% type = 'central' or 'raw'
#% calculate all the moments of order N given samples X and corr. weight w
#[r, n] = size(X);
#% r is the number of samples
#% n is the dimension of the system
#
#if N == 1
#    W = repmat(w, 1, n);
#    m = sum(W.*X, 1);
#    return
#end
#
#combos = GenerateIndex(n, (N + 1) * ones(1, n));
#combos(find(combos == (N + 1))) = 0;
#
#y = [];
#for i = 1: 1: length(combos)
#    if sum(combos(i, : )) == N
#     y = vertcat(y, combos(i, :));
#%      x = vertcat(x, wrev(combos(i, :)));
#    end
#end
#y = sortrows(y, -1);
#
#[yy, yyy] = size(y);
#
#
#m = zeros(1, yy);
#switch lower(type)
#    case 'raw'
#        for i = 1: 1: yy
#        p = w;
#           for j = 1: 1: n
#            p = p.*X(:, j).^y(i, j);
#           end
#          m(i)=sum(p);
#        end
#    case 'central'
#        %first calculate the means
#        W=repmat(w,1,n);
#        mu=sum(W.*X,1);
#        MU=repmat(mu,r,1);
#        X=X-MU;
#        for i=1:1:yy
#        p=w;
#        for j=1:1:n
#        p=p.*X(:,j).^y(i,j);
#        end
#         m(i)=sum(p);
#        end
#    otherwise
#        error('are u nuts, can u spell!!!!only raw or central are allowed')
#
#
#end
#
#
#end
#
#
#
## %%
#
#function X=unifpts_trans(X,bdd_low,bdd_up)
#
#n=size(X,2);
#mu=(bdd_low+bdd_up)/2;
#h=-bdd_low+bdd_up;
#for i=1:1:n
#    X(:,i)=(h(i)/2)*X(:,i)+mu(i);
#end
#
#
#
#
#





# %%







# %%






# %%
















# %%





# %%






# %%
