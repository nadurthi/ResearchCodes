# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:37:25 2019

@author: nadur
"""
import numpy as np
import copy 
import uuid
import numba
import ipdb

# numba.set_num_threads(4)
class PolyNDset:
    def __init__(self):
        self.polys = []
    def appendpoly(self,p):
        self.polys.append(p)
        
    def combineWtdSet(self,c):
        dim = int(np.max([pp._dim for pp in self.polys]))
        p = PolyND(dim=dim)
        for i in range(len(self.polys)):
            p=p+self.polys[i]*c[i]
        return p
    
    def __getitem__(self,i):
        if i<len(self.polys):
            return self.polys[i]
        else:
            return None
    def __len__(self):
        return len(self.polys)
        
class PolyND:
    def __init__(self,var='x',dim=2,nterms=1,varstates=None):
        self.ID = uuid.uuid4()
        self._dim = dim
        self.var = var
        if varstates is None:
            self.varstates  = [var+str(i+1) for i in range(self._dim)]
        else:
            self.varstates = varstates
            
        self.c=np.zeros(nterms)
        self.powers=np.zeros((nterms,dim)).astype(int)
    
    
        
    
    @staticmethod
    def constantpoly(c,var='x',dim=2):
        P = PolyND(var=var,dim=dim,nterms=1)
        P.c = c
        return P
    @staticmethod
    def emptypoly(var='x',dim=2):
        P = PolyND(var=var,dim=dim,nterms=1)
        return P
    
    def makeCopy(self,newID=True):
        return self.copy(newID=newID)
    
    def copy(self,newID=True):
        P=copy.deepcopy(self)
        if newID:
            P.ID = uuid.uuid4()
        return P
    
    def appendMonomials(self,c,pows):
        self.c = np.hstack([self.c,c])
        self.powers = np.vstack([self.powers,pows])
    

        
    def simplify(self):
        # ipdb.set_trace()
        # indx, =  np.where((np.sum(self.powers,axis=1) > 0)) #.all(axis=1).nonzero()
        # self.c = self.c[indx]
        # self.powers = self.powers[indx]
                 
        ind = np.argsort( np.sum(self.powers,axis=1) )
        self.powers = self.powers[ind,:]
        self.c = self.c[ind]
        # totalpow = np.sum(P.powers,axis=1)
        uniqpows = np.unique(self.powers, axis=0)
        c=[]
        
        for tp in uniqpows:
            # ipdb.set_trace()
            indx, = np.where((self.powers == tp).all(axis=1))
            c.append(np.sum(self.c[indx]))
        self.c = np.array(c)
        self.powers = uniqpows
        
        
                 
    def __add__(self,other):
        return add_polyND(self,other)
    
    def __sub__(self,other):
        return sub_polyND(self,other)
        
    def __mul__(self,other):
        if isinstance(other, self.__class__):
            return multiply_polyND(self,other)
        if np.isscalar(other):
            pp = self.copy(newID=True)
            pp.multiplyScalar(other)
            return pp
        
    def multiplyScalar(self,a):
        self.c =  a*self.c
    
    def addScalar(self,a):
        self.c=np.hstack([self.c,a])
        self.powers = np.vstack([self.powers,np.zeros(self._dim)])
        self.simplify()
        
    def __call__(self,x):
        return evaluate_polyND2(self.c,self.powers,x)
        
    def evaluate(self,x):
        return evaluate_polyND2(self.c,self.powers,x)
    

        
    @property
    def nterms(self):
        assert len(self.c) == self.powers.shape[0], "length of coeff. and powers not the same"
        return len(self.c)
    
    @property
    def dim(self):
        return self.powers.shape[1]
    
    def __str__(self):
        print('dim = ',self.dim,' nterms = ',self.nterms)
        ss = []
        for i in range(self.nterms):
            if self.c[i]>0:
                gg= '+'+str(np.abs(self.c[i]))
            if self.c[i]<0:
                gg= '-'+str(np.abs(self.c[i]))
            if self.c[i]==0:
                continue
            
            for j in range(self.dim):
                gg = gg+ '('+self.varstates[j]+ ')' + '^' + str(int(self.powers[i,j])) 
            ss.append( gg)
        return ' '.join(ss)
        
    def __repr__(self):
        return self.__str__()
    

class Exppdf_polyND:
    def __init__(self,P,mean=None,cov=None):
        self.P=P.copy()
        self.mean = mean
        self.cov = cov
        
    def pdf(self,x):
        return np.exp(self.P.evaluate(x))
    
    def mean(self,recompute=False):
        pass
    def cov(self,recompute=False):
        pass
    
    


def multiply_polyND(P1,P2):


    
    P=None
    if P1.var != P2.var:
        P=PolyND(var=P1.var+P2.var,dim=P1.dim+P2.dim,nterms=1,varstates = P1.varstates+P2.varstates)
    
    
    if P1.dim > P2.dim:
        if P is None:
            P=PolyND(var=P1.var,dim=P1.dim ,nterms=1,varstates = P1.varstates)
        pows1 = P1.powers
        e = P1.powers.shape[0]-P2.powers.shape[0]
        pows2 = np.hstack([P2.powers,np.zeros(P2.powers.shape[0],e).reshape(-1,1)])
        
    if P1.dim < P2.dim:
        if P is None:
            P=PolyND(var=P2.var,dim=P2.dim ,nterms=1,varstates = P2.varstates)
        pows2 = P2.powers
        e = P2.powers.shape[0]-P1.powers.shape[0]
        pows1 = np.hstack([P1.powers,np.zeros(P1.powers.shape[0],e).reshape(-1,1)])
        
    if P1.dim == P2.dim:
        if P is None:
            P=PolyND(var=P2.var,dim=P2.dim ,nterms=1,varstates = P2.varstates)
        pows1 = P1.powers
        pows2 = P2.powers
     

    for i in range(P1.nterms):
        c = P1.c[i]*P2.c
        pows = pows1[i]*pows2
        P.appendMonomials(c,pows)

    
    
    
    P.simplify()
    return P


#%%
def add_polyND(P1,P2):

 
    
    P=None
    if P1.var != P2.var:
        P=PolyND(var=P1.var+P2.var,dim=P1.dim+P2.dim,nterms=1,varstates = P1.varstates+P2.varstates)
    
    
    if P1.dim > P2.dim:
        if P is None:
            P=PolyND(var=P1.var,dim=P1.dim ,nterms=1,varstates = P1.varstates)
        pows1 = P1.powers
        e = P1.powers.shape[0]-P2.powers.shape[0]
        pows2 = np.hstack([P2.powers,np.zeros(P2.powers.shape[0],e).reshape(-1,1)])
        
    if P1.dim < P2.dim:
        if P is None:
            P=PolyND(var=P2.var,dim=P2.dim ,nterms=1,varstates = P2.varstates)
        pows2 = P2.powers
        e = P2.powers.shape[0]-P1.powers.shape[0]
        pows1 = np.hstack([P1.powers,np.zeros(P1.powers.shape[0],e).reshape(-1,1)])
        
    if P1.dim == P2.dim:
        if P is None:
            P=PolyND(var=P2.var,dim=P2.dim ,nterms=1,varstates = P2.varstates)
        pows1 = P1.powers
        pows2 = P2.powers
     

    P.varstates
    P.appendMonomials(np.hstack([P1.c,P2.c]),np.vstack([pows1,pows2]))

    
    
    
    P.simplify()
    return P

def sub_polyND(P1,P2):


    P=None
    if P1.var != P2.var:
        P=PolyND(var=P1.var+P2.var,dim=P1.dim+P2.dim,nterms=1,varstates = P1.varstates+P2.varstates)
    
    
    if P1.dim > P2.dim:
        if P is None:
            P=PolyND(var=P1.var,dim=P1.dim ,nterms=1,varstates = P1.varstates)
        pows1 = P1.powers
        e = P1.powers.shape[0]-P2.powers.shape[0]
        pows2 = np.hstack([P2.powers,np.zeros(P2.powers.shape[0],e).reshape(-1,1)])
        
    if P1.dim < P2.dim:
        if P is None:
            P=PolyND(var=P2.var,dim=P2.dim ,nterms=1,varstates = P2.varstates)
        pows2 = P2.powers
        e = P2.powers.shape[0]-P1.powers.shape[0]
        pows1 = np.hstack([P1.powers,np.zeros(P1.powers.shape[0],e).reshape(-1,1)])
        
    if P1.dim == P2.dim:
        if P is None:
            P=PolyND(var=P2.var,dim=P2.dim ,nterms=1,varstates = P2.varstates)
        pows1 = P1.powers
        pows2 = P2.powers
     


    P.appendMonomials(np.hstack([P1.c,-P2.c]),np.vstack([pows1,pows2]))

    
    
    
    P.simplify()
    return P

# %%
fastmath = False
cache = True
@numba.jit(nopython=True,cache=cache,fastmath=fastmath)
def prodcols(A):
    a=np.ones(A.shape[0])
    for i in range(A.shape[1]):
        a *= A[:,i]
    
    return a

@numba.jit(nopython=True,cache=cache,fastmath=fastmath)
def powerfunc(x,powers):
    return x**powers

@numba.jit(nopython=True,cache=cache,fastmath=fastmath)
def sumup(x):
    a = 0
    for i in range(len(x)):
        a += x[i]
    return a
         
def evaluate_polyND(P,x):


    nt=P.nterms;

    
    if x.ndim ==1:
        n=1
        return np.sum(P.c*np.prod(x**P.powers,axis=1))
    
    f=np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        f[i] = np.sum(P.c*np.prod(x[i]**P.powers,axis=1))    
        
    return f


@numba.jit(nopython=True,cache=cache, parallel=True,nogil=True,fastmath=fastmath)
def evaluate_polyND2(c,powers,x):

    if x.ndim ==1:
        return np.sum(c*prodcols(x**powers))
    
    f=np.zeros(x.shape[0])
    for i in numba.prange(x.shape[0]):
        # ipdb.set_trace()
        f[i] = sumup(c*prodcols(powerfunc(x[i],powers)))    
        
    return f

# %%
# function f=addexpterms(z1,z2,s1,s2)
# % s1 and s2 are sign terms
# % s1*exp(z1)+s2*exp(z2)
# % thres1 = 10;
# % q1=floor(z1/thres1);
# % r1=z1-thres1*q1;
# % 
# % q2=floor(z2/thres1);
# % r2=z2-thres1*q2;
# % 
# % cq = min(q1,q2);
# % z1 = z1 - thres1*cq;
# % z2 = z2 - thres1*cq;

# z = min(z1,z2);
# z1 = z1 - z;
# z2 = z2 - z;

# thres = 30;

# if z1+z2 <= thres
#     p=s1*exp(z1) + s2*exp(z2);
#     f = [sign(p),log(abs(p))];
#     f(2) = z+f(2); 
# else % abs(z1) > thres && abs(z2) > thres
# %     [z1,z2,s1,s2]
#     f=zeros(1,2);
#     f(1) = sign(s1*z1+s2*z2);
#     if z1==0
#         f(2) = vpa( log( abs(s1+s2*exp(sym(z2))) ) );
#     else
#         f(2) = vpa( z+log( abs(s1*exp(sym(z1) )+s2) ) );
#     end
# end
    
   
# %%
# function M=AddSubMultiply_MatrixOfPolys(M1,M2,op)
# % M1 and M2 are matrix of polys i.e. cells



# if strcmpi(op,'add')
#     M=cell(size(M1));
    
#     if isequal(size(M1),size(M2) )==0
#         disp('dim of two matrix polys not the same')
#         M=NaN;
#         return
#     end
    
#     for i=1:1:size(M1,1)
#         for j=1:1:size(M1,2)
#             M{i,j}=add_sub_polyND(M1{i,j},M2{i,j},'add');
#         end
#     end
    
# elseif strcmpi(op,'sub')
#     M=cell(size(M1));
    
#     if isequal(size(M1),size(M2) )==0
#         disp('dim of two matrix polys not the same')
#         M=NaN;
#         return
#     end
    
#     for i=1:1:size(M1,1)
#         for j=1:1:size(M1,2)
#             M{i,j}=add_sub_polyND(M1{i,j},M2{i,j},'sub');
#         end
#     end
    
# elseif strcmpi(op,'multiply')
#     M=cell(size(M1,1),size(M1,2));
    
#     if size(M1,2)~=size(M2,1)
#         disp('dim of two matrix polys not the same')
#         M=NaN;
#         return
#     end
    
    
#     for i=1:1:size(M1,1)
#         for j=1:1:size(M2,2)
#             M{i,j}=zeros(1,size(M1{1,1},2));
#             for k=1:1:size(M1,2)
#             M{i,j}=add_sub_polyND(M{i,j},  multiply_polyND(M1{i,k},M2{k,j})  ,'add');  
#             end
#         end
#     end
    
    
# end

# %%
# function f=evaluate_polyND(P,x)
# % evaluates the multidim polynomial P at x.
# % care: dim of x has to be compatible with P
# % P: first column is always coeff, rest of the columns are power of the
# % x=x(:)';

# % 
# % P=vpa(P);
# % x=vpa(x);

# % f=evaluate_polyND2(P,x);
# % return

# %%
# nt=size(P,1);
# dim=size(P,2)-1;
# C=P(:,1);
# M=P(:,2:end);

# [xr,xc]=size(x);
# if xr==1 || xc==1
#     if dim==1
#         x=x(:);
#     else
#     x=x(:)';
#     end
# end
# [xr,xc]=size(x);
# if size(P,2)-1 ~= xc
#     error('dim mismatch in evaluate_polyND')
#     f=NaN;
#     return;
# end
# % prod(repmat(x(1,:),nt,1).^M,2)
# f=zeros(size(x,1),1);
# for i=1:1:size(x,1)
# % f(i)=vpa(sum(C.*vpa(prod(vpa(repmat(x(i,:),nt,1).^M),2)) ));
# f(i)=(sum(C.*(prod((repmat(x(i,:),nt,1).^M),2)) ));
# % f(i)=vpa(sum(C.*prod(repmat(sym(x(i,:)),nt,1).^M,2) ));
# end

# % f
# end

#%%


#%%
# function c = get_coeff_NDpoly(P,pws)
# % where pws are the powers exponents
# % get the coefficient of the term with powers pws
# p=0;
# pws_idx=0;
# for j=1:size(P,2)-1
#     p=p+P(:,j+1)*10^(j-1);
#     pws_idx = pws_idx + pws(j)*10^(j-1);
# end

# ind = find(p==pws_idx);
# if isempty(ind)
#     c= 0;
# else
#     c= P(ind,1);
# end

#%%
# function P=get_gauss_quad_poly(m,pcov)
# dim = length(m);
# m=m(:);
# % given a mean and cov get the corresponding poly p(x) 
# % A=inv(P)
# % -0.5*(x-mu)'*A*(x-mu) = -0.5*( x'Ax-2x'Amu+mu'Amu )
# A=inv(pcov);

# % find the new x1 and x2
# Px=cell(dim,1);
# II=eye(dim);
# for i=1:dim
#     Px{i}=[1,II(i,:)];
# end

# %first multiple A with x
# AP=cell(dim,1);
# for i=1:dim
#    p=[0,zeros(1,dim)];
#    for j=1:dim
#        p1=scalar_multiply_polyND(A(i,j),Px{j});
#        p=add_sub_polyND(p,p1,'add');
#    end
#    AP{i}=p;
   
# end
# P1=[0,zeros(1,dim)];
# for i=1:dim
#     p1=multiply_polyND(Px{i},AP{i});
#     P1=add_sub_polyND(P1,p1,'add');
# end
# d=A*m(:);
# P2=[0,zeros(1,dim)];
# for i=1:dim
#     p1=scalar_multiply_polyND(d(i),Px{i});
#     P2=add_sub_polyND(P2,p1,'add');
# end
# P2=scalar_multiply_polyND(-2,P2);
# P=add_sub_polyND(P1,P2,'add');

# c=m'*A*m;
# P=add_sub_polyND([c,zeros(1,dim)],P,'add');
# P=scalar_multiply_polyND(-0.5,P);


# cexp = log(1/sqrt(det(2*pi*pcov)));
# c0 = get_coeff_NDpoly(P,zeros(1,dim));
# P = update_or_insert_coeff_NDpoly(P,zeros(1,dim),c0+cexp);


#%%
# function Pnew=get_partial_polyND(P,xfix,fixedinds)
# % P is the polynomial
# % fixedinds are indicies of x that are fixed to given x value
# % the remaining varaibles are in the exact same order

# if length(xfix)~=length(fixedinds)
#     error('xfix and fixedinds have to be same length')
# end
# ndim=size(P,2)-1;
# remainids = 1:ndim;
# remainids(fixedinds)=[];

# nt=size(P,1);


# Pnew = zeros(nt,length(remainids));
# Pnew(:,1)=P(:,1);
# for i=1:length(remainids)
#    Pnew(:,1+i) = P(:,1+remainids(i));
# end


# xfix=xfix(:)';
# Pnew(:,1)=Pnew(:,1).*prod(repmat(xfix,nt,1).^P(:,1+fixedinds),2);
# Pnew=simplify_polyND(Pnew);



# end


#%%

# function d=Det_MatrixOfPolys(M)

# polydim=size(M{1,1},2);
# % M is a matrix of polynomaisl
# m=size(M,1);
# if size(M,1)~=size(M,2)
#     disp('Poly Moatrix si not square')
#     d=NaN;
#     return
# end

# if m==2
#     d=  add_sub_polyND(multiply_polyND(M{1,1},M{2,2})  ,  multiply_polyND(M{1,2},M{2,1})  ,'sub')  ;

# elseif m==1
#     d=M{1,1};
    
# elseif m<=2 && m>2  % just use leibniz rule
    
#     P = perms(1:m);

#     II=eye(m);
    
#     d=zeros(1,polydim);
#     for n=1:1:size(P,1)
#         p=[1,zeros(1,polydim-1)];
#         for i=1:1:m
#             p=multiply_polyND( p  ,  M{i,P(n,i)}  );
            
#         end
       
#         d=add_sub_polyND(d,  scalar_multiply_polyND(  det(II(P(n,:),:) ) , p ) ,'add');
        
        
#     end
    
# else

#     d=zeros(1,polydim);
#     for j=1:1:m
#         B=M;
#         B(1,:)=[];
#         B(:,j)=[];
#         d=add_sub_polyND(d,  multiply_polyND( scalar_multiply_polyND(  (-1)^(j+1)  , M{1,j} ),  Det_MatrixOfPolys(B)  ) ,'add');

#     end
    
# end



# %%
# function P=diff_polyND(P,n)
# % differentiate the poly P with respect to the nth variable
# ind=find(P(:,n+1)~=0);
# ind0=find(P(:,n+1)==0);
# P(ind,1)=P(ind,1).*P(ind,n+1);
# P(ind,n+1)=P(ind,n+1)-1;
# P(ind0,1)=0;

# P(find(P(:,1)==0),:)=[];

# P=simplify_polyND(P);
# end


# %%


















# %%



if __name__=="__main__":
    # p = PolyND(var='x',dim=3,nterms=1)
    # numba.set_num_threads(4)
    import time
    N=8
    Np=5
    d=6
    c=np.random.rand(N)
    powers = np.abs(np.random.randint(0,high=30,size=(N,d)))
    x=np.random.rand(Np,d)
    f=evaluate_polyND2(c,powers,x)
    
    N=8000
    Np=150000
    d=6
    c=np.random.rand(N)
    powers = np.abs(np.random.randint(0,high=30,size=(N,d)))
    x=np.random.rand(Np,d)
    st = time.time()
    f=evaluate_polyND2(c,powers,x)
    et = time.time()
    print("time taken is = ",et-st)
    
    