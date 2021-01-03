from ndpoly import polybase as pb
import numpy as np
import numpy.linalg as nplg
import scipy.linalg as sclg

def GenerateIndex(ND,numbasis):
    # %
    # % This function computes all permutations of 1-D basis functions.
    # %
    index = np.arange(numbasis[0]).astype(int).reshape(-1,1) # %short for canonical_0 - first dimension's nodes: this will be loooped through the dimensions
    for ct in range(1,ND): #   %good loop! - over the dimensions
        repel = index.copy() #%REPetition-ELement
        repsize = len(index[:,0]) #;  %REPetition SIZE
        repwith = np.zeros((repsize,1)) #;  %REPeat WITH this structure: initialization
        
        
        for rs in range(1,numbasis[ct]):
            repwith = np.vstack([repwith, np.ones((repsize,1))*rs])    #%update REPeating structure
        
        

        index = np.hstack([np.tile(repel,[numbasis[ct],1]), repwith])    #%update canon0
        


    return index

def MomentOrders(N,nx):
    # % N is the order or  highest degree
    # % nx is the number of variables
    
    if N==0:
        y=np.zeros((1,nx))
        return y
    
    
    combos = GenerateIndex(nx,(N+1)*np.ones(nx).astype(int))
    combos[combos==(N+1)]=0

    
    y=[]
    for i in range(len(combos)):
        if np.sum(combos[i])==N:
            y.append(combos[i])
    y=np.vstack(y)
    # % y=sortrows(y,-1);
    ind=np.argsort(np.sqrt(np.sum(y**2,axis=1)))
    y=y[ind]
    
    return y

def Basis_polyND(n,m):
    # % n is dim
    # % m is the order upto
    # X=np.zeros((2,n))
    # w=np.zeros((2,1))
    c=[1]
    powers = [np.zeros(n)]

    if m==0:
        P = pb.PolyND(var='x',dim=n,nterms=1)
        P.appendMonomials(np.array(c),np.vstack(powers))
        P.simplify()
        polyset = pb.PolyNDset()
        polyset.appendpoly(P)
        return polyset
    
    # k=2;
    for N in range(1,m+1):
        y=MomentOrders(N,n);
        for j in range(y.shape[0]):
            c.append(1)
            powers.append(y[j])

        
    polyset = pb.PolyNDset()
    for i in range(len(c)):
        P = pb.PolyND(var='x',dim=n,nterms=1)
        P.appendMonomials(c[i],powers[i])
        P.simplify()
        polyset.appendpoly(P)
        
    return polyset


class BasisFit:
    def __init__(self,dim,maxNorder = 4):

        self.dim = dim
        self.maxNorder = maxNorder
        self.Pf={}
        for i in range(1,maxNorder+1):
            self.Pf[i]=Basis_polyND(self.dim,i)
            
    def solve_lstsqrs(self,x,fx,order):
        A=np.zeros((x.shape[0],len(self.Pf[order])))
        B=np.zeros(x.shape[0])
        for i in range(A.shape[1]):
            A[:,i] = self.Pf[order][i].evaluate(x)
        B = fx
        
        c,residue,rank,s = nplg.lstsq(A,B)
        polyfit = self.Pf[order].combineWtdSet(c)
        return c,polyfit
    
# %%
# function A=evaluate_BasisPolyMatrix(Pfs,X)
# % A is of dim (N,Nbasis) 
# [N,dim] = size(X);
# Nbasis = length(Pfs);
# A=zeros(N,Nbasis);

# for ib=1:1:Nbasis
#     P=Pfs{ib};
    
# end

# end

# %%
# function f=evaluate_MatrixOfPolys(Pfs,x)
# % evaluates the multidim polynomial P at x.
# % care: dim of x has to be compatible with P
# % P: first column is always coeff, rest of the columns are power of the
# % x=x(:)';

# f=zeros(size(Pfs));

# for i=1:1:size(f,1)
#     for j=1:1:size(f,2)
#         f(i,j)=evaluate_polyND(Pfs{i,j},x);
#     end
# end

# end

# %%
# function S=evaluate_PolyforLargetSetX(P,X)
# % A is of dim (N,Nbasis) 
# [N,dim] = size(X);

# S=0;
# for i=1:1:size(P,1)
#     const = P(i,1);
#     mono = P(i,2:end);
#     S=S+const*prod(X.^repmat(mono,N,1),2);
# end

# end

# %%

if __name__=="__main__":
    # numbasis=np.array([3,3])
    ND = 3
    numbasis=np.array([3]*ND)
    combos = GenerateIndex(ND,numbasis)
    
    y=MomentOrders(5,ND)
    print(y)
    
    polyset = Basis_polyND(3,4)
    