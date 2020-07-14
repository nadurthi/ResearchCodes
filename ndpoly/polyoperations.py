# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:22:24 2019

@author: nadur
"""


#%%
def multiply_polyND(P1,P2):
# =============================================================================
# % c1=max(abs(P1(:,1)));
# % c2=max(abs(P2(:,1)));
# % 
# % P1(:,1)=P1(:,1)/c1;
# % P2(:,1)=P2(:,1)/c2;
# 
# % function to multiply 2 multidimensional polynomials
# %first column is always coeff, rest of the columns are power of the
# %variables
# 
# % [P1,P2]=equalizeDIM_polyND(P1,P2);
# 
# =============================================================================
    m1 = P1.nterms
    d1 = P1.dim
    m2 = P2.nterms
    d2 = P2.dim
    
[m1,d]=size(P1);
[m2,d]=size(P2);
P=zeros(m1*m2,d);
k=1;

if m1<m2   % then rep m1
    
    for i=1:1:m2
        P(m1*(i-1)+1:i*m1,1)=(P1(:,1).*repmat(P2(i,1),m1,1));
        P(m1*(i-1)+1:i*m1,2:end)=(P1(:,2:end)+repmat(P2(i,2:end),m1,1));
        
    end
    
    
else
    
    for i=1:1:m1
        P(m2*(i-1)+1:i*m2,1)=(P2(:,1).*repmat(P1(i,1),m2,1));
        P(m2*(i-1)+1:i*m2,2:end)=(P2(:,2:end)+repmat(P1(i,2:end),m2,1));
        
    end
    
end


P=simplify_polyND(P);

end
#%%
function P=add_sub_polyND(P1,P2,type)
% add P1+P2 or sub P1-P2

switch lower(type)
    case 'add'
        P=[P1;P2];
        P=simplify_polyND(P);
    case 'sub'
        P2(:,1)=-P2(:,1); % as first col are coefficients
        P=[P1;P2];
        P=simplify_polyND(P);
    otherwise
        disp('type not known in add_sub_poly')
        P=NaN;
end


# %%
function f=addexpterms(z1,z2,s1,s2)
% s1 and s2 are sign terms
% s1*exp(z1)+s2*exp(z2)
% thres1 = 10;
% q1=floor(z1/thres1);
% r1=z1-thres1*q1;
% 
% q2=floor(z2/thres1);
% r2=z2-thres1*q2;
% 
% cq = min(q1,q2);
% z1 = z1 - thres1*cq;
% z2 = z2 - thres1*cq;

z = min(z1,z2);
z1 = z1 - z;
z2 = z2 - z;

thres = 30;

if z1+z2 <= thres
    p=s1*exp(z1) + s2*exp(z2);
    f = [sign(p),log(abs(p))];
    f(2) = z+f(2); 
else % abs(z1) > thres && abs(z2) > thres
%     [z1,z2,s1,s2]
    f=zeros(1,2);
    f(1) = sign(s1*z1+s2*z2);
    if z1==0
        f(2) = vpa( log( abs(s1+s2*exp(sym(z2))) ) );
    else
        f(2) = vpa( z+log( abs(s1*exp(sym(z1) )+s2) ) );
    end
end
    
   
# %%
function M=AddSubMultiply_MatrixOfPolys(M1,M2,op)
% M1 and M2 are matrix of polys i.e. cells



if strcmpi(op,'add')
    M=cell(size(M1));
    
    if isequal(size(M1),size(M2) )==0
        disp('dim of two matrix polys not the same')
        M=NaN;
        return
    end
    
    for i=1:1:size(M1,1)
        for j=1:1:size(M1,2)
            M{i,j}=add_sub_polyND(M1{i,j},M2{i,j},'add');
        end
    end
    
elseif strcmpi(op,'sub')
    M=cell(size(M1));
    
    if isequal(size(M1),size(M2) )==0
        disp('dim of two matrix polys not the same')
        M=NaN;
        return
    end
    
    for i=1:1:size(M1,1)
        for j=1:1:size(M1,2)
            M{i,j}=add_sub_polyND(M1{i,j},M2{i,j},'sub');
        end
    end
    
elseif strcmpi(op,'multiply')
    M=cell(size(M1,1),size(M1,2));
    
    if size(M1,2)~=size(M2,1)
        disp('dim of two matrix polys not the same')
        M=NaN;
        return
    end
    
    
    for i=1:1:size(M1,1)
        for j=1:1:size(M2,2)
            M{i,j}=zeros(1,size(M1{1,1},2));
            for k=1:1:size(M1,2)
            M{i,j}=add_sub_polyND(M{i,j},  multiply_polyND(M1{i,k},M2{k,j})  ,'add');  
            end
        end
    end
    
    
end

# %%
function f=evaluate_polyND(P,x)
% evaluates the multidim polynomial P at x.
% care: dim of x has to be compatible with P
% P: first column is always coeff, rest of the columns are power of the
% x=x(:)';

% 
% P=vpa(P);
% x=vpa(x);

% f=evaluate_polyND2(P,x);
% return

%%
nt=size(P,1);
dim=size(P,2)-1;
C=P(:,1);
M=P(:,2:end);

[xr,xc]=size(x);
if xr==1 || xc==1
    if dim==1
        x=x(:);
    else
    x=x(:)';
    end
end
[xr,xc]=size(x);
if size(P,2)-1 ~= xc
    error('dim mismatch in evaluate_polyND')
    f=NaN;
    return;
end
% prod(repmat(x(1,:),nt,1).^M,2)
f=zeros(size(x,1),1);
for i=1:1:size(x,1)
% f(i)=vpa(sum(C.*vpa(prod(vpa(repmat(x(i,:),nt,1).^M),2)) ));
f(i)=(sum(C.*(prod((repmat(x(i,:),nt,1).^M),2)) ));
% f(i)=vpa(sum(C.*prod(repmat(sym(x(i,:)),nt,1).^M,2) ));
end

% f
end

#%%
function f=evaluate_polyND2(P,x)
% evaluates the multidim polynomial P at x.
% care: dim of x has to be compatible with P
% P: first column is always coeff, rest of the columns are power of the
% x=x(:)';

% 
% P=vpa(P);
% x=vpa(x);

nt=size(P,1);
C=P(:,1);
M=P(:,2:end);

[xr,xc]=size(x);
if xr==1 || xc==1
    x=x(:)';
end
[xr,xc]=size(x);
if size(P,2)-1 ~= xc
    error('dim mismatch in evaluate_polyND')
    f=NaN;
    return;
end
% prod(repmat(x(1,:),nt,1).^M,2)
f=zeros(size(x,1),1);
for i=1:1:size(x,1)
    S = [C,repmat(x(i,:),nt,1).^M];
    S = sym(S);
    f(i) = vpa(sum(prod(S,2)));
    
% f(i)=vpa(sum(C.*vpa(prod(vpa(repmat(x(i,:),nt,1).^M),2)) ));
% f(i)=(sum(C.*(prod((repmat(x(i,:),nt,1).^M),2)) ));
% f(i)=vpa(sum(C.*prod(repmat(sym(x(i,:)),nt,1).^M,2) ));
end

% f
end

#%%
function c = get_coeff_NDpoly(P,pws)
% where pws are the powers exponents
% get the coefficient of the term with powers pws
p=0;
pws_idx=0;
for j=1:size(P,2)-1
    p=p+P(:,j+1)*10^(j-1);
    pws_idx = pws_idx + pws(j)*10^(j-1);
end

ind = find(p==pws_idx);
if isempty(ind)
    c= 0;
else
    c= P(ind,1);
end

#%%
function P=get_gauss_quad_poly(m,pcov)
dim = length(m);
m=m(:);
% given a mean and cov get the corresponding poly p(x) 
% A=inv(P)
% -0.5*(x-mu)'*A*(x-mu) = -0.5*( x'Ax-2x'Amu+mu'Amu )
A=inv(pcov);

% find the new x1 and x2
Px=cell(dim,1);
II=eye(dim);
for i=1:dim
    Px{i}=[1,II(i,:)];
end

%first multiple A with x
AP=cell(dim,1);
for i=1:dim
   p=[0,zeros(1,dim)];
   for j=1:dim
       p1=scalar_multiply_polyND(A(i,j),Px{j});
       p=add_sub_polyND(p,p1,'add');
   end
   AP{i}=p;
   
end
P1=[0,zeros(1,dim)];
for i=1:dim
    p1=multiply_polyND(Px{i},AP{i});
    P1=add_sub_polyND(P1,p1,'add');
end
d=A*m(:);
P2=[0,zeros(1,dim)];
for i=1:dim
    p1=scalar_multiply_polyND(d(i),Px{i});
    P2=add_sub_polyND(P2,p1,'add');
end
P2=scalar_multiply_polyND(-2,P2);
P=add_sub_polyND(P1,P2,'add');

c=m'*A*m;
P=add_sub_polyND([c,zeros(1,dim)],P,'add');
P=scalar_multiply_polyND(-0.5,P);


cexp = log(1/sqrt(det(2*pi*pcov)));
c0 = get_coeff_NDpoly(P,zeros(1,dim));
P = update_or_insert_coeff_NDpoly(P,zeros(1,dim),c0+cexp);


#%%
function Pnew=get_partial_polyND(P,xfix,fixedinds)
% P is the polynomial
% fixedinds are indicies of x that are fixed to given x value
% the remaining varaibles are in the exact same order

if length(xfix)~=length(fixedinds)
    error('xfix and fixedinds have to be same length')
end
ndim=size(P,2)-1;
remainids = 1:ndim;
remainids(fixedinds)=[];

nt=size(P,1);


Pnew = zeros(nt,length(remainids));
Pnew(:,1)=P(:,1);
for i=1:length(remainids)
   Pnew(:,1+i) = P(:,1+remainids(i));
end


xfix=xfix(:)';
Pnew(:,1)=Pnew(:,1).*prod(repmat(xfix,nt,1).^P(:,1+fixedinds),2);
Pnew=simplify_polyND(Pnew);



end


#%%

function d=Det_MatrixOfPolys(M)

polydim=size(M{1,1},2);
% M is a matrix of polynomaisl
m=size(M,1);
if size(M,1)~=size(M,2)
    disp('Poly Moatrix si not square')
    d=NaN;
    return
end

if m==2
    d=  add_sub_polyND(multiply_polyND(M{1,1},M{2,2})  ,  multiply_polyND(M{1,2},M{2,1})  ,'sub')  ;

elseif m==1
    d=M{1,1};
    
elseif m<=2 && m>2  % just use leibniz rule
    
    P = perms(1:m);

    II=eye(m);
    
    d=zeros(1,polydim);
    for n=1:1:size(P,1)
        p=[1,zeros(1,polydim-1)];
        for i=1:1:m
            p=multiply_polyND( p  ,  M{i,P(n,i)}  );
            
        end
       
        d=add_sub_polyND(d,  scalar_multiply_polyND(  det(II(P(n,:),:) ) , p ) ,'add');
        
        
    end
    
else

    d=zeros(1,polydim);
    for j=1:1:m
        B=M;
        B(1,:)=[];
        B(:,j)=[];
        d=add_sub_polyND(d,  multiply_polyND( scalar_multiply_polyND(  (-1)^(j+1)  , M{1,j} ),  Det_MatrixOfPolys(B)  ) ,'add');

    end
    
end



# %%
function P=diff_polyND(P,n)
% differentiate the poly P with respect to the nth variable
ind=find(P(:,n+1)~=0);
ind0=find(P(:,n+1)==0);
P(ind,1)=P(ind,1).*P(ind,n+1);
P(ind,n+1)=P(ind,n+1)-1;
P(ind0,1)=0;

P(find(P(:,1)==0),:)=[];

P=simplify_polyND(P);
end


# %%


















