# -*- coding: utf-8 -*-
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import numpy as np
import numpy.linalg as nplg
from uq.gmm import gmmbase as uqgmmbase
from uq.gmm import merger as uqgmmmerg
from uq.gmm import splitter as uqgmmsplit
from utils.math import geometry as utmthgeom
from utils.plotting import geometryshapes as utpltshp
from utils.plotting import surface as utpltsurf
plt.close('all')
from sklearn import mixture


# %%
fovr=1
fovc = np.array([0,0])

xf0 = np.array([1,1])
xf1 = np.array([1,0])
xf2 = np.array([0,1])

Pf0 = np.array([[1,0.3],[0.3,1]])
Pf1 = np.array([[1,0.3],[0.3,1]])
Pf2 = np.array([[1,0.3],[0.3,1]])

gmm = uqgmmbase.GMM.fromlist([xf0,xf1,xf2],[Pf0,Pf1,Pf2],[0.5,0.25,0.25],0)


fig = plt.figure()
ax = fig.add_subplot(111)
X = uqgmmbase.plotGMM2Dcontour(gmm,nsig=1,N=100,rettype='list')
for i in range(len(X)):
    ax.plot(X[i][:,0],X[i][:,1],label=str(i))
    ax.plot(gmm.m(i)[0],gmm.m(i)[1],'ks')

Xfov=utpltshp.getCirclePoints2D(fovc,fovr)
ax.plot(Xfov[:,0],Xfov[:,1],'k--')
ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmm,Ng=50)
# h = plt.contourf(xx,yy,p)
ax.plot_surface(xx,yy,p,alpha=0.8)
for i in range(len(X)):
    ax.plot(X[i][:,0],X[i][:,1],label=str(i))
    mm=gmm.m(i)
    ax.scatter(mm[0],mm[1],0,marker='s',facecolor='k')
Xfov=utpltshp.getCirclePoints2D(fovc,fovr)
ax.plot(Xfov[:,0],Xfov[:,1],'k--')

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# %%



fig = plt.figure()
ax = fig.add_subplot(111)
X = utpltshp.getCovEllipsePoints2D(xf0,Pf0,nsig=1,N=100)
ax.plot(X[:,0],X[:,1])

pdforig = multivariate_normal(xf0, Pf0)
gmm1=uqgmmsplit.splitGaussianUT(xf0,Pf0,w=1,alpha=2)
X2 = uqgmmbase.plotGMM2Dcontour(gmm1,nsig=1,N=100,rettype='list')
for i in range(len(X2)):
    ax.plot(X2[i][:,0],X2[i][:,1],'.-',label=str(i))
    ax.plot(gmm1.m(i)[0],gmm1.m(i)[1],'ks')
    
# Xfov=utpltshp.getCirclePoints2D(fovc,fovr)
# ax.plot(Xfov[:,0],Xfov[:,1],'k--')
ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmm1,Ng=50)
# h = plt.contourf(xx,yy,p)
ax.plot_surface(xx,yy,p,alpha=0.4)
for i in range(len(X2)):
    ax.plot(X2[i][:,0],X2[i][:,1],label=str(i))
    mm=gmm1.m(i)
    ax.scatter(mm[0],mm[1],0,marker='s',facecolor='k')

xx,yy,p = utpltsurf.plotpdf2Dsurf(pdforig,Ng=50)
ax.plot_surface(xx,yy,p,alpha=0.8,cmap=cm.coolwarm)

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# %%

fovr = 1
fovc = np.array([0,0])

fig = plt.figure()
ax = fig.add_subplot(111)
X = utpltshp.getCovEllipsePoints2D(xf0,Pf0,nsig=1,N=100)
ax.plot(X[:,0],X[:,1])
Xfov=utpltshp.getCirclePoints2D(fovc,fovr,N=100)
ax.plot(Xfov[:,0],Xfov[:,1],'k--')
ax.axis('equal')
utmthgeom.isIntersect_circle_cov(fovc,fovr,xf0,Pf0,1)
print(utmthgeom.isIntersect_circle_cov(fovc,fovr,xf0,Pf0,1))

#%% Adaptive splitting 

fovr = 1
fovc = np.array([0,0])
Xfov=utpltshp.getCirclePoints2D(fovc,fovr,N=100)
PD=0.95
gmm = uqgmmbase.GMM.fromlist([xf0,xf1,xf2],[Pf0,Pf1,Pf2],[0.5,0.25,0.25],0)
gmmorig = gmm.makeCopy()

fig = plt.figure()
ax = fig.add_subplot(111)
cnt = 0
while 1:
    
    gmmN = uqgmmbase.GMM(None,None,None,0)
    flg= True
    for idx in range(gmm.Ncomp):
        if utmthgeom.isIntersect_circle_cov(fovc,fovr,gmm.m(idx),gmm.P(idx),1):
            # gmm1=uqgmmsplit.splitGaussianUT(gmm.m(idx),gmm.P(idx),w=gmm.w(idx),alpha=5)
            Xs = np.random.multivariate_normal(gmm.m(idx),gmm.P(idx), 5000)
            y = utmthgeom.isinside_points_circle(Xs,fovc,fovr)
            Xsinside = Xs[y,:]
            Xsoutside = Xs[~y,:]
            
            pdfC = multivariate_normal(gmm.m(idx),gmm.P(idx))
            gmmC = uqgmmbase.GMM(None,None,None,0)
            
            clf = mixture.GaussianMixture(n_components=5, covariance_type='full')
            clf.fit(Xsinside)
            gmmC.appendCompArray(clf.means_,clf.covariances_,clf.weights_,renormalize = True)
            
            clf = mixture.GaussianMixture(n_components=5, covariance_type='full')
            clf.fit(Xsoutside)
            gmmC.appendCompArray(clf.means_,clf.covariances_,clf.weights_,renormalize = True)
            
            gmmC.normalizeWts()
            
            pt=pdfC.pdf(Xs)
            pestc = gmmC.evalcomp(Xs,gmmC.idxs)
            W = nplg.lstsq(pestc.T, pt)
            
            w = W[0]/np.sum(W[0])
            w = w*gmm.w(idx)
            gmmC.setwts(w)
            
            
                                
            gmmN.appendGMM(gmmC)
            flg = False
        else:
            gmmN.appendComp(gmm.m(idx),gmm.P(idx),gmm.w(idx),renormalize = False)
    
    gmm = gmmN
    gmm.normalizeWts()
    cnt += 1
    
    
    print(cnt,gmm.Ncomp)
    ax.cla()
    X2 = uqgmmbase.plotGMM2Dcontour(gmm,nsig=1,N=100,rettype='list')
    for i in range(len(X2)):
        ax.plot(X2[i][:,0],X2[i][:,1],'.-',label=str(i))
    ax.plot(Xfov[:,0],Xfov[:,1],'k--')

    plt.pause(0.1)
    
    if flg is True:
        break
    break

for ii in range(gmm.Ncomp):
    if utmthgeom.isoverlap_circle_cov(fovc,fovr,gmm.m(ii),gmm.P(ii),1,minfrac=70):
        gmm.wts[ii] = gmm.wts[ii]*(1-PD)
        print(ii, " is inside")

gmm.normalizeWts()
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmorig,Ng=50)
ax.plot_surface(xx,yy,p,alpha=0.98,linewidth=1)

ax.plot(Xfov[:,0],Xfov[:,1],'k--')



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(Xfov[:,0],Xfov[:,1],'k--')

xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmm,Ng=50)
ax.plot_surface(xx,yy,p,alpha=0.98,linewidth=1)

