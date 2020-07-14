# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:24:42 2019

@author: nadur
"""

function Pf=Basis_polyND(n,m)
% n is dim
% m is the order upto
X=zeros(2,n);
w=zeros(2,1);
Pf=cell(nchoosek(n+m,m),1);
Pf{1}=[1 zeros(1,n)];
if m==0
    return
end
k=2;
for N=1:1:m
    y=MomentOrders(N,n);
    for j=1:1:size(y,1)
        Pf{k}=[1,y(j,:)];
        k=k+1;
    end
end

# %%
function A=evaluate_BasisPolyMatrix(Pfs,X)
% A is of dim (N,Nbasis) 
[N,dim] = size(X);
Nbasis = length(Pfs);
A=zeros(N,Nbasis);

for ib=1:1:Nbasis
    P=Pfs{ib};
    
end

end

# %%
function f=evaluate_MatrixOfPolys(Pfs,x)
% evaluates the multidim polynomial P at x.
% care: dim of x has to be compatible with P
% P: first column is always coeff, rest of the columns are power of the
% x=x(:)';

f=zeros(size(Pfs));

for i=1:1:size(f,1)
    for j=1:1:size(f,2)
        f(i,j)=evaluate_polyND(Pfs{i,j},x);
    end
end

end

# %%
function S=evaluate_PolyforLargetSetX(P,X)
% A is of dim (N,Nbasis) 
[N,dim] = size(X);

S=0;
for i=1:1:size(P,1)
    const = P(i,1);
    mono = P(i,2:end);
    S=S+const*prod(X.^repmat(mono,N,1),2);
end

end

# %%

