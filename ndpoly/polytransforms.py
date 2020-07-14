# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:29:36 2019

@author: nadur
"""

function S=binomexp_NDpoly(P1,P2,n)
% get the poly (P1+P2)^n
% keyboard
if isempty(P1)==1 && isempty(P2)==0
    if size(P2,1)==1
        S=P2;
        S(1)=S(1)^(n);
        h=S(2:end);
        h = h * (n);
        S(2:end)=h;
    else
        S=binomexp_NDpoly(P2(1,:),P2(2:end,:),n);
    end
    return
end
if isempty(P1)==0 && isempty(P2)==1
    if size(P1,1)==1
        S=P1;
        S(1)=S(1)^(n);
        h=S(2:end);
        h = h * (n);
        S(2:end) = h;
    else
        S=binomexp_NDpoly(P1(1,:),P1(2:end,:),n);
    end
    return
end

dim=size(P1,2)-1;
if n==0
    S= [1,zeros(1,dim)];
    return
end

S=zeros(1,dim+1);
for i=0:n
    s=nchoosek(n,i);
    if size(P1,1)==1
        if n-i==0
            p1=[1,zeros(1,dim)];
        else
            p1=P1;
            p1(1)=p1(1)^(n-i);
            h=p1(2:end);
            h = h * (n-i);
            p1(2:end) = h;
        end
    else
        p1 = binomexp_NDpoly(P1(1,:),P1(2:end,:),n-i);
    end
    
    if size(P2,1)==1
        if i==0
            p2=[1,zeros(1,dim)];
        else
            p2=P2;
            p2(1)=p2(1)^(i);
            h = p2(2:end);
            h = h * (i);
            p2(2:end) = h;
        end
    else
        p2 = binomexp_NDpoly(P2(1,:),P2(2:end,:),i);
    end
    p=multiply_polyND(p1,p2);
    p = scalar_multiply_polyND(s,p);
    S=add_sub_polyND(S,p,'add');
end
S=simplify_polyND(S);



# %%
function M=ConvertMat_2_MatrixOfPolys(A,polydim_withoutcoeff)
% A is a regular matrix... convert into polynomial matrix
d=polydim_withoutcoeff;

M=cell(size(A));

for i=1:1:size(A,1)
    for j=1:1:size(A,2)
        M{i,j}=[A(i,j),zeros(1,d)];  % +1 for the coefficient
    end
end



# %%
function p = get_line_poly(P,a,b)
% here x1 = a(1)*t+b(1)
% x2 = a(2)*t+b(2)
% x3 = a(3)*t+b(3)

% then return a poly in t
dim=size(P,2)-1;
p=0;
for i = 1:size(P,1)
    c=P(i,1);
    p1=[1];
    for j=1:dim
        pc=[a(j),b(j)];
        pc=poly_pow(pc,P(i,j+1));
        p1=conv(p1,pc);
    end
    p1=c*p1;
    
    p=p+p1;
    
end

#%%
