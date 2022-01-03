# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:16:00 2021

@author: nadur
"""


import numpy as np
import numpy.linalg as nplinalg
from numpy.linalg import multi_dot
from scipy.linalg import block_diag

A=np.array([[1,2],[3,4]])
H=np.array([[1,1],[2,2]])

Q0=np.array([[1,0],[0,2]])
Q1=np.array([[1,0],[0,2]])



R1=np.array([[1,0],[0,2]])
R2=np.array([[2,0],[0,1]])


y1=H.dot(A.dot(m0p))+np.sqrt(R1).dot(np.random.randn(2))
A2=A.dot(A)
y2=H.dot(A2.dot(m0p))+np.sqrt(R1).dot(np.random.randn(2))


P0p=np.array([[4,1],[1,5]])
m0p=np.array([1,1])



II=np.identity(2)
ZZ=np.zeros((2,2))
A2=A.dot(A)
HA=H.dot(A)
HA2=H.dot(A2)
C=np.block([[A,II,ZZ,ZZ,ZZ],[A2,A,II,ZZ,ZZ],[HA,H,ZZ,II,ZZ],[HA2,HA,H,ZZ,II]])

D=block_diag(P0p,Q0,Q1,R1,R2)
P=multi_dot([C,D,C.T])

mxx=np.block([A.dot(m0p),A2.dot(m0p)])
myy=np.block([HA.dot(m0p),HA2.dot(m0p)])

yy=np.block([y1,y2])

m1m = A.dot(m0p)
P1m = multi_dot([A,P0p,A.T]) +Q0
S1=multi_dot([H,P1m,H.T])+R1
W1=multi_dot([P1m,H.T,nplinalg.inv(S1)])
Pxy1=P1m.dot(H.T)
m1p=m1m+W1.dot(y1-H.dot(m1m))
# P1p=P1m-multi_dot([W1,H,P1m])
P1p=P1m-multi_dot([Pxy1,nplinalg.inv(S1),Pxy1.T])

m2m = A.dot(m1p)
P2m = multi_dot([A,P1p,A.T])+Q1
S2=multi_dot([H,P2m,H.T])+R2
W2=multi_dot([P2m,H.T,nplinalg.inv(S2)])
Pxy2=P2m.dot(H.T)
m2p=m2m+W2.dot(y2-H.dot(m2m))
# P2p=P2m-multi_dot([W2,H,P2m])
P2p=P2m-multi_dot([Pxy2,nplinalg.inv(S2),Pxy2.T])

Imy = 0.5*np.log(nplinalg.det(P1m)/nplinalg.det(P1p))+0.5*np.log(nplinalg.det(P2m)/nplinalg.det(P2p))

P10=P1m
P20=multi_dot([A,P10,A.T])+Q1
Pxx=np.block([[P10,P10.dot(A.T)],[A.dot(P10),P20]])
G=np.block([[P10.dot(H.T),multi_dot([P10,A.T,H.T])],[multi_dot([A,P10,H.T]),P20.dot(H.T)]])
X=np.block([[multi_dot([H,P10,H.T])+R1,multi_dot([H,P10,A.T,H.T])],[multi_dot([H,A,P10,H.T]),multi_dot([H,P20,H.T])+R2]])

Pxx2=Pxx-multi_dot([G,nplinalg.inv(X),G.T])
mxx2 = mxx+multi_dot([G,nplinalg.inv(X),yy-myy])
Icomp = 0.5*np.log(nplinalg.det(Pxx)/nplinalg.det(Pxx2))

Px1x2y1y2=np.block([[Pxx,G],[G.T,X]])
mx1x2y1y2=np.block([mxx,myy])
#%%

Px1y1 = Px1x2y1y2[np.ix_([0,1,4,5],[0,1,4,5])]
mx1y1=mx1x2y1y2[[0,1,4,5]]

Sa=Px1y1[np.ix_([0,1],[0,1])]
Sc=Px1y1[np.ix_([0,1],[2,3])]
Sb=Px1y1[np.ix_([2,3],[2,3])]

ma = mx1y1[[0,1]]
mb = mx1y1[[2,3]]

mau=ma+multi_dot([Sc,nplinalg.inv(Sb),y1-mb])
Pau=Sa-multi_dot([Sc,nplinalg.inv(Sb),Sc.T])
I11=0.5*np.log(nplinalg.det(Sa)/nplinalg.det(Pau))

#%%
Px2y1y2 = Px1x2y1y2[np.ix_([2,3,4,5,6,7],[2,3,4,5,6,7])]
mx2y1y2 = mx1x2y1y2[[2,3,4,5,6,7]]

Sa=Px2y1y2[np.ix_([0,1],[0,1])]
Sc=Px2y1y2[np.ix_([0,1],[2,3,4,5])]
Sb=Px2y1y2[np.ix_([2,3,4,5],[2,3,4,5])]

ma = mx2y1y2[[0,1]]
mb = mx2y1y2[[2,3,4,5]]

mau2=ma+multi_dot([Sc,nplinalg.inv(Sb),yy-mb])
Pau2=Sa-multi_dot([Sc,nplinalg.inv(Sb),Sc.T])

0.5*np.log(nplinalg.det(Sa)/nplinalg.det(Pau2))

#%%
Px2y1 = Px1x2y1y2[np.ix_([2,3,4,5],[2,3,4,5])]
mx2y1 = mx1x2y1y2[[2,3,4,5]]

Sa=Px2y1[np.ix_([0,1],[0,1])]
Sc=Px2y1[np.ix_([0,1],[2,3])]
Sb=Px2y1[np.ix_([2,3],[2,3])]

ma = mx2y1[[0,1]]
mb = mx2y1[[2,3]]

mau21=ma+multi_dot([Sc,nplinalg.inv(Sb),y1-mb])
Pau21=Sa-multi_dot([Sc,nplinalg.inv(Sb),Sc.T])

I22=0.5*np.log(nplinalg.det(Pau21)/nplinalg.det(Pau2))

0.5*np.log(nplinalg.det(P2m)/nplinalg.det(P2p))

