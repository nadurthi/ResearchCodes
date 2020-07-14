# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:37:25 2019

@author: nadur
"""
import numpy as np


class PolyND:
    def __init__(self,var='x',dim=2,nterms=1):
        self._dim = dim
        self.var = var
        self._nterms = nterms
        self.c=np.zeros(nterms)
        self.powers=np.zeros((nterms,dim)).astype(int)
    
    @staticmethod
    def constantpoly(c,var='x',dim=2):
        P = PolyND(var=var,dim=dim,nterms=1)
        P.c = c
        return P
    
    def normalize(self):
        self = normalizePolyND(self)
        
    def __add__(self,other):
        print('add')
        print(other)
        
    def __mul__(self,other):
        print('mul')
        print(other)
    
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
            for j in range(self.dim):
                ss.append( str(self.c[i]) + self.var + str(j+1) + '^' + str(self.powers[i,j]) )
        return ' + '.join(ss)
        
P1 = PolyND(dim=2)
P2 = PolyND(dim=2)

print(P1)




def normalizePolyND(P):
    ind = np.argsort( np.sum(P.powers,axis=1) )
    P.powers = P.powers[ind,:]
    P.c = P.c[ind]
    totalpow = np.sum(P.powers,axis=1)
    for tp in np.unique(totalpow):
        pass
    
    
    return P