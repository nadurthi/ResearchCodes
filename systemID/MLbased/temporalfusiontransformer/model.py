# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch

# Gated linear units
class GLU(nn.module):
    def __init__(self,model_state_dim):
        self.super().__init__()
        self.L1 = nn.Linear(model_state_dim, model_state_dim)
        self.L2 = nn.Linear(model_state_dim, model_state_dim)
        self.glupytorch = nn.GLU()
        
    def forward(self,x):
        s=self.L1(x)
        y=self.L2(x)        
        o = self.glupytorch(s,y)
        return o
    
    
#Gated residula network
# GRN is used flexible nonlinear-linear selection
#think about it as ELU
class GRN(nn.module):
    def __init__(self,model_state_in_dim,model_state_out_dim,model_context_state_in_dim,
                 dropout_prob=0.2):
        self.super().__init__()
        self.model_state_in_dim = model_state_in_dim
        self.model_state_out_dim = model_state_out_dim
        self.model_context_state_in_dim=model_context_state_in_dim
    
        
        self.dropout_prob=dropout_prob
        
        self.glu = GLU(model_state_out_dim)
        self.layernorm = nn.LayerNorm()
        self.elu = nn.CELU()
        self.L2 = nn.Linear(model_state_in_dim, model_state_out_dim)
        self.L3 = nn.Linear(model_context_state_in_dim, model_state_out_dim, bias=False)
        
        self.L1 = nn.Linear(model_state_out_dim, model_state_out_dim)
        self.drop_layer = nn.Dropout(p=dropout_prob)
        
    def forward(self,a,c):
        N,dim=a.shape
        if c==0:
            c=torch.zeros((N,self.model_context_state_in_dim),device=a.device)
            
        n2 = self.elu(self.L2(a)+self.L3(c))
        n1 = self.L1(n2)
        dn1=self.drop_layer(n1)
        y = self.layernorm(a+self.glu(dn1) )
        
        return y        
        
#  Variable selection Network
# takes elements in of size model_state_dim. 
# total elements are mX
class VSN(nn.module):
    def __init__(self,model_state_in_dim,model_state_out_dim,model_context_state_in_dim,mX):
        self.super().__init__()
        self.model_state_in_dim = model_state_in_dim
        self.model_state_out_dim = model_state_out_dim
        self.model_context_state_in_dim=model_context_state_in_dim
        self.mX=mX
        
        self.inputGRNS = nn.ModuleList([GRN(model_state_out_dim,model_state_out_dim,dropout_prob=0.2) for i in range(mX)])
        self.concatinputGRN=GRN(mX*model_state_in_dim,mX,model_context_state_in_dim,dropout_prob=0.2)
        self.softmax = nn.Softmax()
    
    def forward(self,X,c):
        Y=torch.zeros_like(X)
        N=X.shape[0]
        # Xdim == N,mX,model_state_dim
        for i in range(self.mX):
            Y[:,i,:]=self.inputGRNS[i](X[:,i,:])
        
        #  concatenated should be [modeldim,modeldim,......,modeldim] mX times
        Xflat = torch.reshape(X, (X.shape[0], -1))
        s=self.concatinputGRN(Xflat,c)
        v=self.softmax(s)
        
        V=v.unsqueeze(0).repeat(N,1,1)
        return torch.sum(X*V,1)   


class TFT(nn.module):
    def __init__(self,static_dim,)
      