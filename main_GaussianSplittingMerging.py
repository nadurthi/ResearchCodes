
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
import time
import utils.timers as utltm
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







#%% Adaptive splitting 

fovr = 1
fovc = np.array([1,1])
Xfov=utpltshp.getCirclePoints2D(fovc,fovr,N=100)
PD=0.8
gmm = uqgmmbase.GMM.fromlist([xf0,xf1,xf2],[Pf0,Pf1,Pf2],[0.5,0.25,0.25],0)
gmmorig = gmm.makeCopy()



    
gmmN = uqgmmbase.GMM(None,None,None,0)
flg= True
for idx in range(gmm.Ncomp):
    if utmthgeom.isIntersect_circle_cov(fovc,fovr,gmm.m(idx),gmm.P(idx),1):
        # gmm1=uqgmmsplit.splitGaussianUT(gmm.m(idx),gmm.P(idx),w=gmm.w(idx),alpha=5)
        with utltm.TimingContext():
            Xs = np.random.multivariate_normal(gmm.m(idx),gmm.P(idx), 1000)
            y = utmthgeom.isinside_points_circle(Xs,fovc,fovr)
            Xsinside = Xs[y,:]
            Xsoutside = Xs[~y,:]
            
            pdfC = multivariate_normal(gmm.m(idx),gmm.P(idx))
            gmmC = uqgmmbase.GMM(None,None,None,0)
            if gmmC.mu is None:
                print("gmmC 1 shaoe ", None)
            else:
                print("gmmC 1 shaoe ", gmmC.mu.shape)
            
            with utltm.TimingContext():
                clf = mixture.GaussianMixture(n_components=5, covariance_type='full')
                clf.fit(Xsinside)
                gmmC.appendCompArray(clf.means_,clf.covariances_,clf.weights_,renormalize = True)
                print("gmmC 2 shaoe ", gmmC.mu.shape)
            
            with utltm.TimingContext():
                clf2 = mixture.GaussianMixture(n_components=5, covariance_type='full')
                clf2.fit(Xsoutside)
                gmmC.appendCompArray(clf2.means_,clf2.covariances_,clf2.weights_,renormalize = True)
                print("gmmC 3 shaoe ", gmmC.mu.shape)
            
            with utltm.TimingContext():
                gmmC.normalizeWts()
                print("gmmC shaoe ", gmmC.mu.shape)
                pt=pdfC.pdf(Xs)
                gmmC.resetwts()
                w=uqgmmbase.optimizeWts(gmmC,Xs,pt)
                gmmC.setwts(gmm.w(idx)*w)
                gmmC.normalizeWts()
        
                            
        gmmN.appendGMM(gmmC)
        flg = False
    else:
        gmmN.appendComp(gmm.m(idx),gmm.P(idx),gmm.w(idx),renormalize = False)




for ii in range(gmmN.Ncomp):
    if utmthgeom.isoverlap_circle_cov(fovc,fovr,gmmN.m(ii),gmmN.P(ii),1,minfrac=70):
        gmmN.wts[ii] = gmmN.wts[ii]*(1-PD)
        print(ii, " is inside")
        
gmmN.normalizeWts()
print("#comp = ",gmmN.Ncomp)

gmmN.pruneByWts(wtthresh=1e-4)
gmmN.normalizeWts()
print("#comp-prunes = ",gmmN.Ncomp)

with utltm.TimingContext():
    gmmNmerged = uqgmmmerg.merge1(gmmN,meanabs=0.5,meanthresfrac=0.5,dh=0.5)
    print("Merged #comp = ",gmmNmerged.Ncomp)

with utltm.TimingContext():
    gmmNmerged = uqgmmmerg.merge1(gmmNmerged,meanabs=0.5,meanthresfrac=0.5,dh=0.5)
    print("Merged #comp = ",gmmNmerged.Ncomp)

with utltm.TimingContext():
    gmmNmerged = uqgmmmerg.merge1(gmmNmerged,meanabs=0.5,meanthresfrac=0.5,dh=0.5)
    print("Merged #comp = ",gmmNmerged.Ncomp)
# Xs = gmmNmerged.random(2000)
# pt = gmmNmerged.pdf(Xs)
# w=uqgmmbase.optimizeWts(gmmNmerged,Xs,pt)
# gmmNmerged.setwts(w)

# Contour Plot of original
fig = plt.figure("original-cont")
ax = fig.add_subplot(111)
X2 = uqgmmbase.plotGMM2Dcontour(gmmorig,nsig=1,N=100,rettype='list')
for i in range(len(X2)):
    ax.plot(X2[i][:,0],X2[i][:,1],'.-',label=str(i))
ax.plot(Xfov[:,0],Xfov[:,1],'k--')
ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.pause(0.1)
    
# Contour Plot of split
fig = plt.figure("splitted-cont")
ax = fig.add_subplot(111)
X2 = uqgmmbase.plotGMM2Dcontour(gmmN,nsig=1,N=100,rettype='list')
for i in range(len(X2)):
    ax.plot(X2[i][:,0],X2[i][:,1],'.-',label=str(i))
ax.plot(Xfov[:,0],Xfov[:,1],'k--')
ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.pause(0.1)


# Contour Plot of merged
fig = plt.figure("merged-cont")
ax = fig.add_subplot(111)
X2 = uqgmmbase.plotGMM2Dcontour(gmmNmerged,nsig=1,N=100,rettype='list')
for i in range(len(X2)):
    ax.plot(X2[i][:,0],X2[i][:,1],'.-',label=str(i))
ax.plot(Xfov[:,0],Xfov[:,1],'k--')
ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.pause(0.1)

        

# Surface Plot of original
fig = plt.figure("original-surf")
ax = fig.add_subplot(111, projection='3d')
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmorig,Ng=50)
ax.plot_surface(xx,yy,p,alpha=0.98,linewidth=1)
ax.plot(Xfov[:,0],Xfov[:,1],'k--')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Surface Plot of splitted
fig = plt.figure("splitted-surf")
ax = fig.add_subplot(111, projection='3d')
ax.plot(Xfov[:,0],Xfov[:,1],'k--')
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmN,Ng=50)
ax.plot_surface(xx,yy,p,alpha=0.98,linewidth=1)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Surface Plot of merged
fig = plt.figure("merged-surf")
ax = fig.add_subplot(111, projection='3d')
ax.plot(Xfov[:,0],Xfov[:,1],'k--')
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmNmerged,Ng=50)
ax.plot_surface(xx,yy,p,alpha=0.98,linewidth=1)
ax.set_xlabel('x')
ax.set_ylabel('y')

