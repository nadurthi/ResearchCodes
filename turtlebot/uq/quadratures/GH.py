#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %%



def GH_pts(mcent,Pcov,Np):
    n=length(mcent);
    
    if Np==1
        x=1;
        w=1;
    
    
    if Np==2
        x=(roots([1,0,-1]));
        B=[1,0,1]';
        A=[1,1;
          x(1),x(2);
          x(1)^2,x(2)^2];
        w=A\B;
    
    
    if Np==3
       x=roots([1,0,-3,0]);
           B=[1,0,1,0,3]';
        A=[1,1,1;
          x(1),x(2),x(3);
          x(1)^2,x(2)^2,x(3)^2;
          x(1)^3,x(2)^3,x(3)^3;
          x(1)^4,x(2)^4,x(3)^4];
        w=A\B;
    
    
    if Np==4
       x=roots([1,0,-6,0,3]);
              B=[1,0,1,0,3,0,15]';
        A=[1,1,1,1;
          x(1),x(2),x(3),x(4);
          x(1)^2,x(2)^2,x(3)^2,x(4)^2;
          x(1)^3,x(2)^3,x(3)^3,x(4)^3;
          x(1)^4,x(2)^4,x(3)^4,x(4)^4;
          x(1)^5,x(2)^5,x(3)^5,x(4)^5;
          x(1)^6,x(2)^6,x(3)^6,x(4)^6];
        w=A\B;
    
    
    if Np==5
       x=roots([1,0,-10,0,15,0]);
                 B=[1,0,1,0,3,0,15,0,105]';
        A=[1,1,1,1,1;
          x(1),x(2),x(3),x(4),x(5);
          x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2;
          x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3;
          x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4;
          x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5;
          x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6;
          x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7;
          x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8];
        w=A\B;
    
    
    if Np==6
       x=roots([1,0,-15,0,45,0,-15]);
                 B=[1,0,1,0,3,0,15,0,105,0,945]';
        A=[1,1,1,1,1,1;
          x(1),x(2),x(3),x(4),x(5),x(6);
          x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2;
          x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3;
          x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4;
          x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5;
          x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6;
          x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7;
          x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8;
          x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9;
          x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10];
        w=A\B;
    
    
    if Np==7
       x=roots([1,0,-21,0,105,0,-105,0]);
                    B=[1,0,1,0,3,0,15,0,105,0,945,0,10395]';
        A=[1,1,1,1,1,1,1;
          x(1),x(2),x(3),x(4),x(5),x(6),x(7);
          x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2,x(7)^2;
          x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3,x(7)^3;
          x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4,x(7)^4;
          x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5,x(7)^5;
          x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6,x(7)^6;
          x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7,x(7)^7;
          x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8,x(7)^8;
          x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9,x(7)^9;
          x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10,x(7)^10;
          x(1)^11,x(2)^11,x(3)^11,x(4)^11,x(5)^11,x(6)^11,x(7)^11;
          x(1)^12,x(2)^12,x(3)^12,x(4)^12,x(5)^12,x(6)^12,x(7)^12];
        w=A\B;
    end
    
    if Np==8
       x=roots([1,0,-28,0,210,0,-420,0,105]);
                       B=[1,0,1,0,3,0,15,0,105,0,945,0,10395,0,135135]';
        A=[1,1,1,1,1,1,1,1;
          x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8);
          x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2,x(7)^2,x(8)^2;
          x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3,x(7)^3,x(8)^3;
          x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4,x(7)^4,x(8)^4;
          x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5,x(7)^5,x(8)^5;
          x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6,x(7)^6,x(8)^6;
          x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7,x(7)^7,x(8)^7;
          x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8,x(7)^8,x(8)^8;
          x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9,x(7)^9,x(8)^9;
          x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10,x(7)^10,x(8)^10;
          x(1)^11,x(2)^11,x(3)^11,x(4)^11,x(5)^11,x(6)^11,x(7)^11,x(8)^11;
          x(1)^12,x(2)^12,x(3)^12,x(4)^12,x(5)^12,x(6)^12,x(7)^12,x(8)^12;
          x(1)^13,x(2)^13,x(3)^13,x(4)^13,x(5)^13,x(6)^13,x(7)^13,x(8)^13;
          x(1)^14,x(2)^14,x(3)^14,x(4)^14,x(5)^14,x(6)^14,x(7)^14,x(8)^14];
        w=A\B;
    
    
    if Np==9
       x=roots([1,0,-36,0,378,0,-1260,0,945,0]);
       B=[1,0,1,0,3,0,15,0,105,0,945,0,10395,0,135135,0,2027025]';
           A=[1,1,1,1,1,1,1,1,1;
          x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9);
          x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2,x(7)^2,x(8)^2,x(9)^2;
          x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3,x(7)^3,x(8)^3,x(9)^3;
          x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4,x(7)^4,x(8)^4,x(9)^4;
          x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5,x(7)^5,x(8)^5,x(9)^5;
          x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6,x(7)^6,x(8)^6,x(9)^6;
          x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7,x(7)^7,x(8)^7,x(9)^7;
          x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8,x(7)^8,x(8)^8,x(9)^8;
          x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9,x(7)^9,x(8)^9,x(9)^9;
          x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10,x(7)^10,x(8)^10,x(9)^10;
          x(1)^11,x(2)^11,x(3)^11,x(4)^11,x(5)^11,x(6)^11,x(7)^11,x(8)^11,x(9)^11;
          x(1)^12,x(2)^12,x(3)^12,x(4)^12,x(5)^12,x(6)^12,x(7)^12,x(8)^12,x(9)^12;
          x(1)^13,x(2)^13,x(3)^13,x(4)^13,x(5)^13,x(6)^13,x(7)^13,x(8)^13,x(9)^13;
          x(1)^14,x(2)^14,x(3)^14,x(4)^14,x(5)^14,x(6)^14,x(7)^14,x(8)^14,x(9)^14;
          x(1)^15,x(2)^15,x(3)^15,x(4)^15,x(5)^15,x(6)^15,x(7)^15,x(8)^15,x(9)^15;
          x(1)^16,x(2)^16,x(3)^16,x(4)^16,x(5)^16,x(6)^16,x(7)^16,x(8)^16,x(9)^16];
        w=A\B;
    
    
    if Np==10
       x=roots([1,0,-45,0,630,0,-3150,0,4725,0,-945]);
       B=[1,0,1,0,3,0,15,0,105,0,945,0,10395,0,135135,0,2027025,0,34459425]';
           A=[1,1,1,1,1,1,1,1,1,1;
          x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10);
          x(1)^2,x(2)^2,x(3)^2,x(4)^2,x(5)^2,x(6)^2,x(7)^2,x(8)^2,x(9)^2,x(10)^2;
          x(1)^3,x(2)^3,x(3)^3,x(4)^3,x(5)^3,x(6)^3,x(7)^3,x(8)^3,x(9)^3,x(10)^3;
          x(1)^4,x(2)^4,x(3)^4,x(4)^4,x(5)^4,x(6)^4,x(7)^4,x(8)^4,x(9)^4,x(10)^4;
          x(1)^5,x(2)^5,x(3)^5,x(4)^5,x(5)^5,x(6)^5,x(7)^5,x(8)^5,x(9)^5,x(10)^5;
          x(1)^6,x(2)^6,x(3)^6,x(4)^6,x(5)^6,x(6)^6,x(7)^6,x(8)^6,x(9)^6,x(10)^6;
          x(1)^7,x(2)^7,x(3)^7,x(4)^7,x(5)^7,x(6)^7,x(7)^7,x(8)^7,x(9)^7,x(10)^7;
          x(1)^8,x(2)^8,x(3)^8,x(4)^8,x(5)^8,x(6)^8,x(7)^8,x(8)^8,x(9)^8,x(10)^8;
          x(1)^9,x(2)^9,x(3)^9,x(4)^9,x(5)^9,x(6)^9,x(7)^9,x(8)^9,x(9)^9,x(10)^9;
          x(1)^10,x(2)^10,x(3)^10,x(4)^10,x(5)^10,x(6)^10,x(7)^10,x(8)^10,x(9)^10,x(10)^10;
          x(1)^11,x(2)^11,x(3)^11,x(4)^11,x(5)^11,x(6)^11,x(7)^11,x(8)^11,x(9)^11,x(10)^11;
          x(1)^12,x(2)^12,x(3)^12,x(4)^12,x(5)^12,x(6)^12,x(7)^12,x(8)^12,x(9)^12,x(10)^12;
          x(1)^13,x(2)^13,x(3)^13,x(4)^13,x(5)^13,x(6)^13,x(7)^13,x(8)^13,x(9)^13,x(10)^13;
          x(1)^14,x(2)^14,x(3)^14,x(4)^14,x(5)^14,x(6)^14,x(7)^14,x(8)^14,x(9)^14,x(10)^14;
          x(1)^15,x(2)^15,x(3)^15,x(4)^15,x(5)^15,x(6)^15,x(7)^15,x(8)^15,x(9)^15,x(10)^15;
          x(1)^16,x(2)^16,x(3)^16,x(4)^16,x(5)^16,x(6)^16,x(7)^16,x(8)^16,x(9)^16,x(10)^16;
          x(1)^17,x(2)^17,x(3)^17,x(4)^17,x(5)^17,x(6)^17,x(7)^17,x(8)^17,x(9)^17,x(10)^17;
          x(1)^18,x(2)^18,x(3)^18,x(4)^18,x(5)^18,x(6)^18,x(7)^18,x(8)^18,x(9)^18,x(10)^18];
        w=A\B;
    
    
    
    
    # %% take tensor product of the points and weights
    X=x;
    W=w;
    for i=1:1:n-1
    [X,W]=tens_prod_vec(X,x,W,w);
    
    
    # %% transforming the points
    A=sqrtm(Pcov);
    for i=1:1:length(W)
        X(i,:)=(A*X(i,:)'+mcent)';
    
    




#%%
# function[index, xint, wint] = Pcomb(ND, numquad, xinteg, winteg)
# index = (1: numquad(1))';  %short for canonical_0 - first dimension's nodes: this will be loooped through the dimensions
# for ct = 2: ND % good loop! - over the dimensions
#     repel = index; % REPetition - ELement
#     repsize = length(index(:, 1));  % REPetition SIZE
#     repwith = ones(repsize, 1); % REPeat WITH this structure: initialization
#     for rs = 2: numquad(ct)
#         repwith = [repwith; ones(repsize, 1) * rs]; % update REPeating structure
#     end
#     index = [repmat(repel, numquad(ct), 1), repwith]; % update canon0
# end
# index = index;

# xint = xinteg(index(:, 1), 1);
# wint = winteg(index(:, 1), 1);
# for j = 2: ND % good loop! - run through the number of dimensions!
#     xint = [xint, xinteg(index(:, j), j)];
#     wint = wint.*winteg(index(:, j), j);
# end


# # %%


# function[phi, phip, InnerProd, weigh, xind, wind] = GramSch(n, m, Np)

# %
# % This function implements the GramSchmidt procedure to generate orthogonal
# % polynomials to a given weight function and compute corresponding inner
# % product
# %
# % Input Variables:
# % n is the desired order of polynomials
# % m is the order of weight function(pdf), m = -2 coreesponds to std.
# % Gaussian, m = -1 corresponds to uniform and m > -1 corresponds to GLOMAP
# % weight functions
# % Np is number of integration points
# %
# % Output Variables:
# %  phip is n x N matrix of polynomial values at prescribed points, xp
# %  InnerProd is n x n x n tensor of basis function inner product.
# % weigh is N x 1 vector of pdf evaluations.


# syms x real


# %%
# % Weight function Expression
# %%
# if m == -2
#     weig = 1 / (2 * pi) * exp(-0.5 * x ^ 2);
#     lb = -inf; ub = inf;
#     [xind, wind] = hermitequad(Np, 0);
# else if m == -1
#         weig = 1 / 2;
#         [xind, wind] = lgwt(Np, -1, 1); lb = -1; ub = 1;
#     else
#         weig = GLOMAP(m,x); lb = -1; ub = 1;
#     end
# end

# %%
# % Compute Orthogonal Polynomials
# %%
# phi(2,1) = x; phi(1,1)=1;

# for ct = 1:n+1
#     psi(ct) = x.^(ct-1);
# end

# for ct = 3:n+1
#     sump = 0;
#     for k = 1: ct-1
#         sump = sump + int(psi(ct)*phi(k)*weig,'x',lb,ub)/int(phi(k)*phi(k)*weig,'x',lb,ub)*phi(k);
#     end
#     phi(ct,1) = psi(ct)-sump;
# end

# % F  = [x,phi];
# % matlabFunction(F,'file','BasisFunc.m')

# for ct = 1:n+1
#     InnerProd(:,:,ct) = (double(int(phi*phi'*phi(ct,1)*weig,'x',lb,ub)));
# end

# xp = xind;


# for ct = 1:length(xp)
#     x = xp(ct);
#     phip(:,ct) = eval(phi);
#     if m==-1
#         weigh(ct,1) = 1/2;
#     else
#     weigh(ct,1) = eval(weig);
#     end
# end


# function weig = GLOMAP(m,x)
# sumw = 0;
# for k = 0:m
#     facm = factorial(m); facmk = factorial(m-k);  fack = factorial(k);
#     sumw = sumw+(-1)^k/(2*m-k+1)*facm/(fack*facmk)*abs(x).^(m-k);
# end
# weig = 1-abs(x).^(m+1)*sumw*factorial(2*m+1)/factorial(m)^2*(-1)^m;















# # %%

# function [xint,wint] = GH_points(mcent,Pcov,Np)




# if Np==1
#     xint=zeros(1,length(mcent));
#     wint=1;
#     return
# end
# %
# % Author: Puneet Singla.
# % Last Modified: October 1st, 2010.
# %
# % This function computes the orthogonal polynomials and their inner
# % products in n-dim parameter space.
# %
# % Input Variables:
# % P is the covariance matrix
# % mcent is the center of the Gaussian component
# % Np number of integration points along each direction
# %
# % Output Variables:
# %
# % xint is Ninteg x ND matrix of quadrature points
# % wint Ninteg x 1 vector of quadrature weights.
# %

# % keyboard
# [U,S,V] = svd(Pcov); S = diag(S);

# if det(U-V) >= 1e-12
#     disp('Covariance Matrix should be Symmteric');
#     return
# end

# ND  =size(Pcov,1); % state dimension

# for ct = 1: ND

#         [xintg,wintg]=HermiteQuad(Np,0,S(ct)); xintg = xintg(:); wintg = wintg(:);
#         xind(:,ct) = xintg; wind(:,ct) = wintg;


# end




# %%
# % ND Integration points and corresponding weights
# %%

# index = GenerateIndex(ND,Np*ones(1,ND));
# xint = xind(index(:,1),1); % ND quadrature points
# wint = wind(index(:,1),1); % ND pdf

# for j = 2:ND %good loop! - run through the number of dimensions!
#     xint = [xint, xind(index(:,j),j)];
#     wint = wint.*wind(index(:,j),j);


# end
# xint = xint'; xint = U*xint+repmat(mcent,1,size(xint,2));xint = xint';














# # %%

# function [xint,wint] = get_colocation(Ninteg, xl, xu)

# % Ninteg = [5 5];
# % xl = [-10 -1];
# % xu = [10 1];


# ND = length(xl);

# for ct = 1:ND
#     [xint, wint] = lgwt(Ninteg(ct),xl(ct),xu(ct)); % Generate Collocation Points
#     xinteg(:,ct) = xint(:);
#     winteg(:,ct) = wint(:);
# end

# clear xint wint

# [index,xint,wint] = Pcomb(ND,Ninteg,xinteg,winteg);
# wint=wint/sum(wint);
# xint = xint;








# # %%
# function [xint,wint,pw] = GenerateQuadPoints(m,ND,Np)

# %
# % Author: Puneet Singla.
# % Last Modified: June 19, 2009.
# %
# % This function computes the orthogonal polynomials and their inner
# % products in n-dim parameter space.
# %
# % Input Variables:
# % N is the maximum order of polynomials along each direction and m detrmines the pdf.
# % m = -2 corresponds to Gaussin pdf of Zero mean and variance 1
# % m = -1 corresponds to uniform pdf over [-1,1]
# % m >=0 corresponds to GLOMAP pdfs
# % ND is number of uncertain parameters, i.e., our basis function will lie in
# % ND dim space.
# % Np number of integration points along each direction
# %
# % Output Variables:
# %
# % xint is Ninteg x ND matrix of quadrature points
# % wint,pw are Ninteg x 1 vectors of quadrature weights and pdf evaluations, respectively.
# %


# for ct = 1: ND
#     if m(ct) == -2
#         [xintg,wintg]=hermitequad(Np(ct),0); xintg = xintg(:); wintg = wintg(:);
#         xind(:,ct) = xintg; wind(:,ct) = wintg;
#         weigh(:,ct) = 1/(2*pi)*exp(-0.5*xintg.^2);


#     else if m(ct) == -1

#             %[xintu,wintu]=lgwt(Np(ct),-1,1); xintu = xintu(:); wintu = wintu(:);
#             [xintu,wintu]=ClenshawCurtis(Np(ct),-1,1); xintu = xintu(:); wintu = wintu(:);
#             xind(:,ct) = xintu; wind(:,ct) = 1/2*wintu;
#             weigh(:,ct) = 1/2*ones(size(xintu));


#         else
#              lb = -1; ub = 1;
#             [xintgl,wintgl]=ClenshawCurtis(Np(ct),lb,ub);

#             weig = GLOMAP(m(ct),xintgl);
#             xind(:,ct) = xintgl; wind(:,ct) = wintgl.*weig(:);
#             weigh(:,ct) = weig(:);
#         end
#     end
# end



# %%
# % ND Integration points and corresponding weights
# %%

# index = GenerateIndex(ND,Np);
# xint = xind(index(:,1),1); % ND quadrature points
# pw = weigh(index(:,1),1); % ND quadrature weights
# wint = wind(index(:,1),1); % ND pdf

# for j = 2:ND %good loop! - run through the number of dimensions!
#     xint = [xint, xind(index(:,j),j)];
#     wint = [wint, wind(index(:,j),j)];
#     pw = pw.*weigh(index(:,j),j);

# end


# function weig = GLOMAP(m,x)
# sumw = 0;
# for k = 0:m
#     facm = factorial(m); facmk = factorial(m-k);  fack = factorial(k);
#     sumw = sumw+(-1)^k/(2*m-k+1)*facm/(fack*facmk)*abs(x).^(m-k);
# end
# weig = 1-abs(x).^(m+1).*sumw*factorial(2*m+1)/factorial(m)^2*(-1)^m;










# # %%
# function [xint,wint] = GenerateQuadPoints(Pcov,mcent,Np)

# %
# % Author: Puneet Singla.
# % Last Modified: October 1st, 2010.
# %
# % This function computes the orthogonal polynomials and their inner
# % products in n-dim parameter space.
# %
# % Input Variables:
# % P is the covariance matrix
# % mcent is the center of the Gaussian component
# % Np number of integration points along each direction
# %
# % Output Variables:
# %
# % xint is Ninteg x ND matrix of quadrature points
# % wint Ninteg x 1 vector of quadrature weights.
# %


# [U,S,V] = svd(Pcov); S = diag(S);

# if det(U-V) >= 1e-12
#     disp('Covariance Matrix should be Symmteric');
#     return
# end

# ND  =size(Pcov,1); % state dimension

# for ct = 1: ND

#         [xintg,wintg]=HermiteQuad(Np,0,S(ct)); xintg = xintg(:); wintg = wintg(:);
#         xind(:,ct) = xintg; wind(:,ct) = wintg;


# end




# %%
# % ND Integration points and corresponding weights
# %%

# index = GenerateIndex(ND,Np*ones(1,ND));
# xint = xind(index(:,1),1); % ND quadrature points
# wint = wind(index(:,1),1); % ND pdf

# for j = 2:ND %good loop! - run through the number of dimensions!
#     xint = [xint, xind(index(:,j),j)];
#     wint = wint.*wind(index(:,j),j);


# end
# xint = xint'; xint = U*xint+repmat(mcent,1,size(xint,2));xint = xint';




# # %%

# function index = GenerateIndex(ND,numbasis)
# %
# % This function computes all permutations of 1-D basis functions.
# %
# index = (1:numbasis(1))';  %short for canonical_0 - first dimension's nodes: this will be loooped through the dimensions
# for ct = 2:ND    %good loop! - over the dimensions
#     repel = index; %REPetition-ELement
#     repsize = length(index(:,1));  %REPetition SIZE
#     repwith = ones(repsize,1);  %REPeat WITH this structure: initialization
#     for rs = 2:numbasis(ct)
#         repwith = [repwith; ones(repsize,1)*rs];    %update REPeating structure
#     end
#     index = [repmat(repel,numbasis(ct),1), repwith];    %update canon0
# end
# index = index;




# # %%








# # %%









# # %%











# # %%
