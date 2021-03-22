# %% logging
import loggerconfig as logconf
logger = logconf.getLogger(__name__)

logger.info('Info log message')
logger.debug('debug message')
logger.error('error example')
logger.verbose('verbose log message')
logger.warning('warn message')
logger.critical('critical message')
logger.timing('timing message',{'funcName':"funny",'funcArgstr':"None",'timeTaken':3.33})

# %% imports
import os
import pickle as pkl
from random import shuffle
import matplotlib
# try:
#     matplotlib.use('TkAgg')
# except:
# matplotlib.use('nbAgg')
import matplotlib.pyplot as plt
colmap = plt.get_cmap('gist_rainbow')

import os
from matplotlib import cm
from scipy.linalg import block_diag
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import numpy as np
import numpy.linalg as nplg
from uq.gmm import gmmbase as uqgmmbase
from uq.gmm import merger as uqgmmmerg
from uq.gmm import splitter as uqgmmsplit
from physmodels import motionmodels as phymm
from physmodels.twobody import twobody as phytbp
from physmodels import sensormodels as physm
from physmodels import sensormodels_fov as physmfov
from uq.uqutils import recorder as uqrecorder
from physmodels import targets as phytarg
from uq.uqutils import metrics as uqmetrics
from utils.math import geometry as utmthgeom
from uq.uqutils import simmanager as uqsimmanager
from utils.plotting import geometryshapes as utpltshp
from utils.plotting import surface as utpltsurf
from uq.quadratures import cubatures as quadcub
from uq.gmm import merger as uqgmmmerg
from uq.gmm import splitter as uqgmmsplit   
from uq.gmm import gmmsplitteralgos as uqgmmsplitalgos
import uq.quadratures.cubatures as uqcb
import uq.filters.kalmanfilter as uqkf
from uq.filters import sigmafilter as uqfsigf
from uq.filters import gmm as uqgmmf
from sklearn import mixture
import scipy.optimize as scopt #block_diag
from uq.transformers import transforms as uqtransf
import collections as clc
from uq.stats import moments as uqstat
import physmodels.twobody.constants as tbpconst
from physmodels.twobody import tbpcoords
from ndpoly import polybasis as plybss
earthconst = tbpconst.PlanetConstants()
tbpdimconvert = tbpcoords.DimensionalConverter(earthconst)

# %% file-level properties

runfilename = __file__
metalog="""
AAS 2020 MOC vs GMM paper simulations
Author: Venkat
Date: June 4 2020

Comparing the GMM vs MOC for the satellite problem
"""

t0 = tbpdimconvert.true2can_time(0)
tf = tbpdimconvert.true2can_time(300*60*60)
dt = tbpdimconvert.true2can_time(36*60*60)

simmanger = uqsimmanager.SimManager(t0=t0,tf=tf,dt=dt,dtplot=dt/10,
                                  simname="AAS-2020-GMMvsMOC",savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()


# %%
x0 = np.array([7000,2000,2000,-2,7.5,3.5])
P0=block_diag(0.001**2,0.001**2,0.001**2,0.00001**2,0.00001**2,0.00001**2)



x0[0:3]=x0[0:3]*tbpdimconvert.trueX2normX
x0[3:6]=x0[3:6]*tbpdimconvert.trueV2normV
P0[0:3,0:3]=P0[0:3,0:3]*tbpdimconvert.trueX2normX**2
P0[3:6,3:6]=P0[3:6,3:6]*tbpdimconvert.trueV2normV**2

gmm = uqgmmbase.GMM.fromarray1Comp(x0,P0,0)
gmm0 = gmm.makeCopy()

x0orb = tbpcoords.cart2classorb(x0,1)
x0orb 

Nmc = 10000
Xmc0=np.random.multivariate_normal(x0,P0, Nmc )
pmc = multivariate_normal(mean=x0,cov=P0).pdf(Xmc0)
# Emc=[]
# for i in range(Xmc0.shape[0]):
#     x0orb=tbpcoords.cart2classorb(Xmc0[i,:],1)
#     Emc.append(x0orb)

uqstat.MeanCov(Xmc0,np.ones(Nmc)/Nmc)

    
tbpmodel = phytbp.TBP6DFnG(mu=1)
Xtruth=tbpmodel.integrate(simmanger.tvec,x0)
Xplot = tbpmodel.integrate(simmanger.tvecplot,x0)


fig = plt.figure("TBP")
ax = fig.add_subplot(111,label='nominal', projection='3d')
ax.set_title("tbp 1")
ax.plot(Xplot[:,0],Xplot[:,1],Xplot[:,2])
plt.pause(0.1)

Xmc = tbpmodel.integrate_batch(simmanger.tvec,Xmc0)

Xgh,wgh = quadcub.GH_points(x0, 0.8**2*P0, 5)
uqstat.MeanCov(Xgh,wgh)
Xghbatch = tbpmodel.integrate_batch(simmanger.tvec,Xgh)
pgh = multivariate_normal(mean=x0,cov=P0).pdf(Xgh)
# E=[]
# for i in range(Xgh.shape[0]):
#     x0orb=tbpcoords.cart2classorb(Xgh[i,:],1)
#     E.append(x0orb)
    

xpdatatransf = uqtransf.XProbdataSet()


plybsfitter = plybss.BasisFit(6,maxNorder = 4)




for t,tk,dt in simmanger.iteratetimesteps():
    print (t,tk,dt)
    
    
    XXgh,ppgh = xpdatatransf.fit_apply_normalize_0I(Xghbatch[:,tk+1,:],pgh)
    XXmc,ppmc = xpdatatransf.applyForward(Xmc[:,tk+1,:],pmc)
    c,polyfit=plybsfitter.solve_lstsqrs(XXgh,np.log(ppgh+np.finfo(float).eps),4)
    
    # XXmc,ppmc = xpdatatransf.fit_apply_normalize_0I(Xmc[:,tk,:],pmc)
    # XXgh,ppgh = xpdatatransf.applyForward(Xghbatch[:,tk,:],pgh)
    # c,polyfit=plybsfitter.solve_lstsqrs(XXmc,np.log(ppmc+np.finfo(float).eps),4)
    
    fig = plt.figure("TBP MC")
    if len(fig.axes) ==0:
        ax = fig.add_subplot(111,label='mc', projection='3d')
    else:
        ax = fig.axes[0]
    ax.cla()
    ax.set_title("time tk = %f"%(tk+1,) )    
    ax.plot(XXmc[:,1],XXmc[:,1],XXmc[:,2],'b.')
    plt.pause(0.5)
    # k = f'{t+dt:06.2f}'.replace('.','-')
    simmanger.savefigure(fig, ['Eg1','MCnormalized'], str(int(tk+1))+'.png',data=[XXmc,t+dt,tk+1])
    
    
    polyfitevalgh = polyfit.evaluate(XXgh)
    polyfitpdfgh = np.exp(polyfitevalgh)
    polyfitevalmc = polyfit.evaluate(XXmc)
    polyfitpdfmc = np.exp(polyfitevalmc)
    # p75 = np.percentile(ppmc,80)
    relerror = 100*np.abs(polyfitpdfmc-ppmc)/ppmc
    relerror = relerror[relerror<np.percentile(relerror,90)]
    print("polyfit max min mean relerror= ",np.max(relerror),np.min(relerror),np.mean(relerror))
    
    fig = plt.figure("TBP poly-fit hist k = %d"%(tk+1,))
    if len(fig.axes) ==0:
        ax = fig.add_subplot(111,label='hist')
    else:
        ax = fig.axes[0]
    ax.cla()
    ax.set_title("time tk = %f"%(tk+1,) )   
    ax.set_xlabel("% rel. error")
    ax.hist(relerror,bins=100)
    # k = f'{t:06.2f}'.replace('.','-')
    simmanger.savefigure(fig, ['Eg1','Histograms'], 'polyfit_'+str(int(tk+1))+'.png',data=[relerror,t+dt,tk+1])
    
    # GMM prop
    def func(x):
        t_eval=[simmanger.t0,t+dt]
        tt,xk1 = tbpmodel.integrate(t_eval,x)
        return xk1
    gmmsplitted = uqgmmsplitalgos.splitGMM_ryanruss(gmm0,func,3,1,7)
    gmmprop = uqgmmbase.nonlinearTransform(gmmsplitted,func)
    
    pdfgmm = gmmprop.pdf(Xmc[:,tk+1,:])
    relerrorgmm = 100*np.abs(pdfgmm-pmc)/pmc
    relerrorgmm = relerrorgmm[relerrorgmm<np.percentile(relerrorgmm,90)]
    print("gmmfit max min mean relerror= ",np.max(relerrorgmm),np.min(relerrorgmm),np.mean(relerrorgmm))
    
    fig = plt.figure("TBP GMM-fit hist k = %d"%(tk+1,))
    if len(fig.axes) ==0:
        ax = fig.add_subplot(111,label='hist')
    else:
        ax = fig.axes[0]
    ax.cla()
    ax.set_title("time tk = %f"%(tk+1,) )   
    ax.set_xlabel("% rel. error")
    ax.hist(relerrorgmm,bins=100)
    # k = f'{t:06.2f}'.replace('.','-')
    simmanger.savefigure(fig, ['Eg1','Histograms'], 'gmmfit_'+str(int(tk+1))+'.png',data=[relerrorgmm,t+dt,tk+1])
    
    
    break

# %%

# eXAMPLE 2: polar to cartesian

def func2(x):
    if x.ndim==1:
        r=x[0]
        th=x[1]
        xk1 = r*np.array([np.cos(th),np.sin(th)])
        return xk1
    else:
        xk1 = np.vstack([x[:,0]*np.cos(x[:,1]),x[:,0]*np.sin(x[:,1])]).T
        return xk1
    
def func2jac(x):
    if x.ndim==1:
        r=x[0]
        th=x[1]
        jac = np.array([[np.cos(th),-r*np.sin(th)],[np.sin(th),r*np.cos(th)]])
        return jac
    else:
        jac=[]
        for i in range(x.shape[0]):
            r=x[i,0]
            th=x[i,1]
            jac.append( np.array([[np.cos(th),-r*np.sin(th)],[np.sin(th),r*np.cos(th)]]) )

        return jac


x0eg2 = np.array([30,np.pi/2])
P0eg2 = np.array([[2**2,0],[0,(15*np.pi/180)**2]])
Nmc = 10000
Xmc0=np.random.multivariate_normal(x0eg2,P0eg2, Nmc )
pmc0 = multivariate_normal(mean=x0eg2,cov=P0eg2).pdf(Xmc0)
Xmcprop = func2(Xmc0)
jacprop = func2jac(Xmc0)
pmcprop = np.array( [pmc0[i]/nplg.det(jacprop[i]) for i in range(Nmc) ] )

fig = plt.figure("MC before")
ax = fig.add_subplot(111)
ax.cla()
ax.plot(Xmc0[:,0],Xmc0[:,1],'r.')
fig = plt.figure("MC after")
ax = fig.add_subplot(111)
ax.cla()
ax.plot(Xmcprop[:,0],Xmcprop[:,1],'b.')


gmm = uqgmmbase.GMM.fromarray1Comp(x0eg2,P0eg2,0)
gmmsplitted = uqgmmsplitalgos.splitGMM_ryanruss(gmm,func2,3,1,4)
gmmprop = uqgmmbase.nonlinearTransform(gmm,func2)
gmmsplitprop = uqgmmbase.nonlinearTransform(gmmsplitted,func2)

Xgh0,wgh0 = quadcub.GH_points(x0eg2, 0.5**2*P0eg2, 8)
pgh0 = multivariate_normal(mean=x0eg2,cov=P0eg2).pdf(Xgh0)
Xghprop = func2(Xgh0)
jacghprop = func2jac(Xgh0)
pghprop = np.array( [pgh0[i]/nplg.det(jacghprop[i]) for i in range(Xghprop.shape[0]) ] )

xpdatatransfeg2 = uqtransf.XProbdataSet()
XXghpropnorm,ppghpropnorm = xpdatatransfeg2.fit_apply_normalize_0I(Xghprop,pghprop)
XXmcpropnorm,ppmcpropnorm = xpdatatransfeg2.applyForward(Xmcprop,pmcprop)
plybsfittereg2 = plybss.BasisFit(2,maxNorder = 8)
c,polyfit=plybsfittereg2.solve_lstsqrs(XXghpropnorm,np.log(ppghpropnorm+np.finfo(float).eps),6)

polyfitevalgh = polyfit.evaluate(XXghpropnorm)
polyfitpdfgh = np.exp(polyfitevalgh)
polyfitevalmc = polyfit.evaluate(XXmcpropnorm)
polyfitpdfmc = np.exp(polyfitevalmc)
# p75 = np.percentile(ppmc,80)
relerror = 100*np.abs(polyfitpdfmc-ppmcpropnorm )/ppmcpropnorm 
relerror = relerror[relerror<np.percentile(relerror,90)]
print("polyfit max min mean relerror= ",np.max(relerror),np.min(relerror),np.mean(relerror))

pdfgmm = gmmsplitprop.pdf(Xmcprop)
relerrorgmm = 100*np.abs(pdfgmm-pmcprop)/pmcprop
relerrorgmm = relerrorgmm[relerrorgmm<np.percentile(relerrorgmm,90)]
print("gmmfit max min mean relerror= ",np.max(relerrorgmm),np.min(relerrorgmm),np.mean(relerrorgmm))


fig = plt.figure("EG2: TBP poly-fit hist ")
if len(fig.axes) ==0:
    ax = fig.add_subplot(111,label='hist')
else:
    ax = fig.axes[0]
ax.cla() 
ax.set_xlabel("% rel. error")
ax.hist(relerror,bins=100)
ax.set_xlabel('rel. error')
plt.pause(0.1)
# simmanger.savefigure(fig, ['Eg2','Histograms'], 'polyfit'+'.png',data=[relerror])
    

fig = plt.figure("EG2: TBP GMM-fit hist ")
if len(fig.axes) ==0:
    ax = fig.add_subplot(111,label='hist')
else:
    ax = fig.axes[0]
ax.cla() 
ax.set_xlabel("% rel. error")
ax.hist(relerrorgmm,bins=100)
ax.set_xlabel('rel. error')
plt.pause(0.1)
# simmanger.savefigure(fig, ['Eg2','Histograms'], 'gmmfit'+'.png',data=[relerrorgmm])
    



fig = plt.figure("gmm-splitted at 0")
ax = fig.add_subplot(111)
# gmmmarg = gmm.marginalize([2,3,4,5],inplace=False)
X2 = uqgmmbase.plotGMM2Dcontour(gmm,nsig=1,N=100,rettype='list')
for i in range(len(X2)):
    ax.plot(X2[i][:,0],X2[i][:,1],'r',label=str(i))


X2 = uqgmmbase.plotGMM2Dcontour(gmmsplitted,nsig=1,N=100,rettype='list')
for i in range(len(X2)):
    ax.plot(X2[i][:,0],X2[i][:,1],'b',label=str(i))

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.pause(0.1)
# simmanger.savefigure(fig, ['Eg2','gmm'], 'gmmfit0'+'.png',data=[gmm,gmmsplitted])



fig = plt.figure("gmm-splitted-transformed")
ax = fig.add_subplot(111)
# gmmmarg = gmm.marginalize([2,3,4,5],inplace=False)
X2 = uqgmmbase.plotGMM2Dcontour(gmmprop,nsig=1,N=100,rettype='list')
for i in range(len(X2)):
    ax.plot(X2[i][:,0],X2[i][:,1],'r',label=str(i))

# gmmmarg = gmmsplitted.marginalize([2,3,4,5],inplace=False)
X2 = uqgmmbase.plotGMM2Dcontour(gmmsplitprop,nsig=1,N=100,rettype='list')
for i in range(len(X2)):
    ax.plot(X2[i][:,0],X2[i][:,1],'b',label=str(i))
ax.plot(Xmcprop[:,0],Xmcprop[:,1],'k.',alpha=0.3)   
# ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.pause(0.1)
# simmanger.savefigure(fig, ['Eg2','gmm'], 'gmmfitprop'+'.png',data=[gmmprop,gmmsplitprop])
 
polyfitevalgh = polyfit.evaluate(XXghpropnorm)
polyfitpdfgh = np.exp(polyfitevalgh)



fig = plt.figure("PolarCart-Surf")
ax = fig.add_subplot(111, projection='3d')
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmprop,Ng=50)
ax.plot_surface(xx,yy,p,color='r',alpha=0.6,linewidth=1) 
XX = np.hstack([xx.reshape(-1,1),yy.reshape(-1,1)  ])
polyfitxx = polyfit.evaluate(XX)
polyfitxxpdf = np.exp(polyfitxx)

xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmsplitprop,Ng=50)
ax.plot_surface(xx,yy,p,color='b',alpha=0.6,linewidth=1) 
ax.plot(Xmcprop[:,0],Xmcprop[:,1],'k.',alpha=0.3)  


fig = plt.figure("PolarCart-contour-prop")
ax = fig.add_subplot(111)
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmprop,Ng=50)
ax.contour(xx, yy, p, levels=10,colors='r')
xx,yy,p = uqgmmbase.plotGMM2Dsurf(gmmsplitprop,Ng=50)
ax.contour(xx, yy, p,levels=10, colors='b')
ax.plot(Xmcprop[:,0],Xmcprop[:,1],'k.',alpha=0.3)

#%%
from uq.stats import pdfs as uqstpdf
X=np.linspace(-5,5,200)
rs,w,sig=uqgmmsplit.splitGaussian1D_ryanruss(5,1)
fig = plt.figure("ComponentPlot")
ax = fig.add_subplot(111)
Y=uqstpdf.gaussianPDF1D(X,0,1)
plt.plot(X,Y)
for i in range(len(rs)):
    Y=uqstpdf.gaussianPDF1D(X,rs[i],sig[i]**2)
    plt.plot(X,w[i]*Y,'--')
    
# %%
simmanger.finalize()

simmanger.save(metalog, mainfile=runfilename, x0=x0, P0=P0,x0eg2=x0eg2, P0eg2=P0eg2,
                Nmc=Nmc,)
