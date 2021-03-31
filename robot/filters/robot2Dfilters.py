import numpy as np
import numpy.linalg as nplnalg
import scipy.linalg as sclnalg
from uq.gmm import merger as uqgmmmerg
from uq.gmm import splitter as uqgmmsplit  
from physmodels import sensormodels as physm
from uq.gmm import gmmbase as uqgmmbase
from utils.math import geometry as utmthgeom
from uq.quadratures import cubatures as uqcub
import scipy.linalg as sclnalg #block_diag
import scipy.optimize as scopt #block_diag
from scipy.stats import multivariate_normal
from sklearn import mixture

import utils.timers as utltm
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from uq.stats import pdfs as uqstpdf
from uq.information import distance as uqinfodis
# fig1 = plt.figure("Debug plot before split")
# ax1 = fig1.add_subplot(111,label='ax1')


# fig2 = plt.figure("Debug plot after split")
# ax2 = fig2.add_subplot(111,label='ax2')

# fig3 = plt.figure("Debug plot after meas update robot 0")
# ax3 = fig3.add_subplot(111,label='ax3')

# fig4 = plt.figure("Debug plot after meas update robot 1")
# ax4 = fig4.add_subplot(111,label='ax4')

def plotgmmtarget(fig,ax,gmmu,xktruth,t,robots,posstates):
    plt.figure(fig.number)
    ax.cla()
    ax.set_title("time step = %f"%(t,) )
    robots[0].mapobj.plotmap(ax)
    for r in robots:
        r.plotrobot(ax)
        
    # xktruth = targetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',t+dt)
            
    gmmupos = uqgmmbase.marginalizeGMM(gmmu,posstates)
    XX = uqgmmbase.plotGMM2Dcontour(gmmupos,nsig=1,N=100,rettype='list')
    for cc in range(gmmupos.Ncomp):
        ax.plot(XX[cc][:,0],XX[cc][:,1],c='g')
        ax.annotate("wt: "+str(gmmupos.w(cc))[:5],gmmupos.m(cc)[0:2],gmmupos.m(cc)[0:2]-3,color='g',fontsize='x-small')
        
    ax.plot(xktruth[:,0],xktruth[:,1],linestyle='--',c='g')
    ax.plot(xktruth[-1,0],xktruth[-1,1],c='g',marker='*')
    # ax.annotate(targetset[i].targetName,xktruth[-1,0:2],xktruth[-1,0:2]+2,color=colors[i],fontsize='x-small')

# 
    ax.axis('equal')
    ax.set(xlim=(-15, 115), ylim=(-15, 115))
    plt.pause(0.1)
    plt.ion()
    plt.show()
        
def measUpdateFOV_randomsplitter(t,dt,robots,target,Zk,Targetfilterer,infoconfig,
                                 updttarget=True,
                                 splitterConfig = uqgmmsplit.splitterConfigDefault,
                                 mergerConfig=uqgmmmerg.mergerConfigDefault,
                                 computePriorPostDist=False):
    """
    - First, get components with nsig intersecting FOV
    - Split the components into inside-outside
    - if zk is not None
        - If comp-mean are inside, do regular measurement update
        - If comp-mean is outside, reduce by 1-PD
    - if zk is None:
        - if comp mean is inside, reduce by 1-PD
        - if comp mean is outside, increase by PD
    """
    
    
    # xktruth = target.groundtruthrecorder.getvar_uptotime_stacked('xtk',t)
    
    # plotgmmtarget(fig1,ax1,target.gmmfk,xktruth,t,robots,target.posstates)
    
    gmmfkpos = uqgmmbase.marginalizeGMM(target.gmmfk,target.posstates)
    # gmmuk = uqgmmbase.GMM(None,None,None,0)
    gmmuk = target.gmmfk.makeCopy()
    for j in range(len(robots)):
        # do robot by robot
        fovr = robots[j].sensormodel.FOVradius
        fovc = robots[j].sensormodel.xc
        gmmNew = uqgmmbase.GMM(None,None,None,0)
        gmmfkpos = uqgmmbase.marginalizeGMM(gmmuk,target.posstates)
        for i in range(gmmuk.Ncomp):
            if utmthgeom.isoverlap_circle_cov(fovc,fovr,gmmfkpos.m(i),
                                              gmmfkpos.P(i),
                                              splitterConfig.nsig,
                                              minfrac=splitterConfig.minfrac):
                Xs = np.random.multivariate_normal(gmmuk.m(i),gmmuk.P(i), splitterConfig.Nmc)
                y = utmthgeom.isinside_points_circle(Xs[:,target.posstates],fovc,fovr)
                
                Xinsidepts = Xs[y,:]
                Xoutsidepts = Xs[~y,:]
                
                if Xinsidepts.shape[0]==0 or Xoutsidepts.shape[0]==0:
                    gmmNew.appendComp(gmmuk.m(i),gmmuk.P(i),gmmuk.w(i),renormalize = False)
                else:
                    gmmC = uqgmmbase.GMM(None,None,None,0)
                    if Xinsidepts.shape[0]>5:
                        Xinsidepts = np.unique(np.vstack(Xinsidepts), axis=0)
            
                        clf = mixture.GaussianMixture(n_components=splitterConfig.NcompIN, covariance_type='full')
                        clf.fit(Xinsidepts)
                        gmmC.appendCompArray(clf.means_,clf.covariances_,clf.weights_,renormalize = True)
                    
                    if Xoutsidepts.shape[0]>5:
                        Xoutsidepts = np.unique(np.vstack(Xoutsidepts), axis=0)
                        
                        clf2 = mixture.GaussianMixture(n_components=splitterConfig.NcompOUT, covariance_type='full')
                        clf2.fit(Xoutsidepts)
                        gmmC.appendCompArray(clf2.means_,clf2.covariances_,clf2.weights_,renormalize = True)
                    
                    gmmC.normalizeWts()
                    if gmmC.Ncomp > 0:
                        pdfC = multivariate_normal(gmmuk.m(i),gmmuk.P(i))
                        pt=pdfC.pdf(Xs)
                        gmmC.resetwts()
                        w=uqgmmbase.optimizeWts(gmmC,Xs,pt)
                        w=w/np.sum(w)
                        gmmC.setwts(gmmuk.w(i)*w)
                        # gmmC.normalizeWts()
                    
                    gmmNew.appendGMM(gmmC)
                    
            else:
                gmmNew.appendComp(gmmuk.m(i),gmmuk.P(i),gmmuk.w(i),renormalize = False)
            
            
        
        gmmNew.normalizeWts()
        gmmNew.pruneByWts(wtthresh=splitterConfig.wtthreshprune,renormalize = True)                                
        
        gmmuk = gmmNew.makeCopy()
    
    # plotgmmtarget(fig2,ax2,gmmuk,xktruth,t,robots,target.posstates)
    
    # if gmmuk.Ncomp>5:
    #     print("lots of comps")
        
    for j in range(len(robots)):
        # measurement update
        fovr = robots[j].sensormodel.FOVradius
        fovc = robots[j].sensormodel.xc
        TP = robots[j].sensormodel.TP
        TN = robots[j].sensormodel.TN
        FP = robots[j].sensormodel.FP
        FN = robots[j].sensormodel.FN
        for i in range(gmmuk.Ncomp):
            # fovr = robots[j].sensormodel.FOVradius
            # fovc = robots[j].sensormodel.xc
            
            yy = utmthgeom.isinside_points_circle(gmmuk.m(i)[target.posstates],fovc,fovr)
            params={}
            w = gmmuk.w(i)
            if yy == True and Zk[j] is not None: # mean is in FOV and zk also in FOV
                xu, Pu, mz, R, Pxz, Pz, K, pdfz, likez = Targetfilterer.modefilterer.measUpdate(t, dt, gmmuk.m(i), gmmuk.P(i), robots[j].sensormodel, Zk[j], **params)
                # print("Mean is IN, and Meas is IN: original w= ",w," updated w= ",TP*likez*w)
                gmmuk.updateComp(i,m=xu,P=Pu,w=TP*likez*w)
            elif yy == False and Zk[j] is not None: # mean is outdie FOV and zk is in FOV
                xu, Pu, mz, R, Pxz, Pz, K, pdfz, likez = Targetfilterer.modefilterer.measUpdate(t, dt, gmmuk.m(i), gmmuk.P(i), robots[j].sensormodel, Zk[j], **params)    
                # print("Mean is OUT, and Meas is IN: original w= ",w," updated w= ",FP*likez*w)
                gmmuk.updateComp(i,w=FP*likez*w)
            elif yy == True and Zk[j] is None: # mean is in FOV and zk is None(outside)
                # print("Mean is IN, and Meas is OUT: original w= ",w," updated w= ",FN*w)
                gmmuk.updateComp(i,w=FN*w)
            elif yy == False and Zk[j] is None: # mean is outside FOV and zk is None(outside)
                # print("Mean is OUT, and Meas is OUT: original w= ",w," updated w= ",TN*w)
                gmmuk.updateComp(i,w=TN*w)
            
            
            else:
                raise Exception("Option for meas update unknonw")
        
        gmmuk.normalizeWts()
        # print("j = ",j, " the wts are: ",gmmuk.wts)
        
        # gmmuk.pruneByWts(wtthresh=mergerConfig.wtthreshprune,renormalize = True)
        # if j==0:
        #     plotgmmtarget(fig3,ax3,gmmuk.makeCopy(),xktruth,t,robots,target.posstates)
        # if j==1:
        #     plotgmmtarget(fig4,ax4,gmmuk.makeCopy(),xktruth,t,robots,target.posstates)
    
    
        
    if computePriorPostDist is True:
        
        # gmmupos = uqgmmbase.marginalizeGMM(gmmuk,target.posstates)
        # m,P = gmmupos.meanCov()
        # u,v = nplnalg.eig(P)
        # if np.max(np.sqrt(u))<5:
        #     dd=0
        # else:
        #     
        dd=uqinfodis.hellingerDistGMM(gmmNew,gmmuk)
        # mpr,Pr = gmmNew.meanCov()
        # mps,Ps = gmmuk.meanCov()
        # detpr = nplnalg.det(Pr)
        # detps = nplnalg.det(Ps)
        # if np.allclose(detpr, detps, rtol=1e-07, atol=1e-09):
        #     dd=0
        # else:
        #     dd = np.log(detpr/detps)
            
        if t not in target.context:
            target.context[t]={'info': dd}    
        else:
            target.context[t]['info'] = dd    
    
    gmmukcollapsed = gmmuk.collapseGMM()
    X1 = gmmukcollapsed.random(1000)
    X2 = gmmuk.random(1000)
    XX = np.vstack([X1,X2])
    pc = gmmukcollapsed.pdf(XX)+1
    po = gmmuk.pdf(XX)+1
    if np.percentile(100*np.abs(pc-po)/po,75)<25:
        gmmuk = gmmukcollapsed
        
    if mergerConfig.doMerge is True:
        
        
        if mergerConfig.alorithm == "EMfixedComps":
            if gmmuk.Ncomp > mergerConfig.fixedComps:
                
                X=gmmuk.random(mergerConfig.fixedCompsNmc)
                if target.gmmfk.Ncomp>=mergerConfig.fixedComps:
                    weights_init = np.ones(mergerConfig.fixedComps)/mergerConfig.fixedComps
                    means_init = target.gmmfk.mu[:mergerConfig.fixedComps].copy()
                    precisions_init = target.gmmfk.covars[:mergerConfig.fixedComps].copy()
                    for ii in range(mergerConfig.fixedComps):
                        precisions_init[ii] = nplnalg.inv(precisions_init[ii])
                    clf = mixture.GaussianMixture(n_components=mergerConfig.fixedComps, covariance_type='full',
                                              weights_init=weights_init,means_init=means_init,precisions_init=precisions_init)
                else:
                    clf = mixture.GaussianMixture(n_components=mergerConfig.fixedComps, covariance_type='full')
                clf.fit(X)
                gmmuk = uqgmmbase.GMM(None,None,None,0)
                gmmuk.appendCompArray(clf.means_,clf.covariances_,clf.weights_,renormalize = True)
        # gmmuk = uqgmmmerg.merge1(gmmuk,mergerConfig=mergerConfig)

        
    gmmuk.pruneByWts(wtthresh=mergerConfig.wtthreshprune,renormalize = True)    
        
    xfk,Pfk = gmmuk.meanCov()
    if updttarget:
        target.setTargetFilterStageAsPosterior()
        target.updateParams(currt=t,gmmfk=gmmuk,xfk=xfk,Pfk=Pfk)
        
    
    return gmmuk

def myresampling(q):
# %Systematic Resampling or Deterministic Resampling
    N = len(q)
    c = np.cumsum(q)
    J = np.zeros(N)
    i = 0
    u1 = np.random.rand()/N;
    
    for j in range(N):
        u = u1 + j/N
        while u>c[i]:
            i = i + 1

        J[j] = i;
    
    return J

# with utltm.TimingContext("Resampling: "):
    #     W=np.cumsum(np.hstack([0,wmc]))
    #     C = np.random.rand(Nmc)
    #     counts,bins = np.histogram(C, bins=W)
    #     XX = []
    #     PP = np.random.randn(Xmc.shape[1],Xmc.shape[1])
    #     PP = 0.01*np.matmul(PP,PP.T)
        
    #     for i in range(len(counts)):
    #         if counts[i]>0:
    #             x=np.random.multivariate_normal(Xmc[i],PP, counts[i] )
    #             XX.append(x)
    #     Xmc_resample = np.vstack(XX)
    # PP = np.random.rand(Xmc.shape[1],Xmc.shape[1])
    # PP = 0.01*np.matmul(PP,PP.T)
    # Neff = 1 / sum(w_new**2)
    # if Neff < 0.5 * Nmc:
        
def measUpdateFOV_PFGMM(t,dt,robots,target,Zk,Targetfilterer,infoconfig,
                                 updttarget=True,
                                 splitterConfig = uqgmmsplit.splitterConfigDefault,
                                 mergerConfig=uqgmmmerg.mergerConfigDefault,
                                 computePriorPostDist=False):
    """

    """
    
    
    # xktruth = target.groundtruthrecorder.getvar_uptotime_stacked('xtk',t)
    
    # plotgmmtarget(fig1,ax1,target.gmmfk,xktruth,t,robots,target.posstates)
    
    # gmmfkpos = uqgmmbase.marginalizeGMM(target.gmmfk,target.posstates)
    # gmmuk = uqgmmbase.GMM(None,None,None,0)
    # print("-------UPDATE--------")
    Nmc = 5000
    Xmc = target.gmmfk.random(Nmc)
    
        
        
    wmc = np.ones(Nmc)/Nmc
    wmc_old = wmc.copy()       
    with utltm.TimingContext("PF update: ",printit = False):
        for j in range(len(robots)):
            # measurement update
            fovr = robots[j].sensormodel.FOVradius
            fovc = robots[j].sensormodel.xc
            TP = robots[j].sensormodel.TP
            TN = robots[j].sensormodel.TN
            FP = robots[j].sensormodel.FP
            FN = robots[j].sensormodel.FN
            R = robots[j].sensormodel.measNoise(t, dt, Xmc[0])
            yy = utmthgeom.isinside_points_circle(Xmc[:,target.posstates],fovc,fovr)
            # print("infrac = ",np.sum(yy)/len(yy), "outfrac = ",1-np.sum(yy)/len(yy))
            if np.all(yy==True) or np.all(yy==False):
                continue
            
            # win = wmc[yy]
            # Xmcin = Xmc[yy,:]
            
            # wout = wmc[~yy]
            # Xmcout = Xmc[~yy,:]
                
            if Zk[j] is not None: # there is a measurement
                
                Zmk = robots[j].sensormodel.evalBatchNoFOV(t, dt, Xmc[yy][:,target.posstates],posstates=target.posstates)
                likez = uqstpdf.gaussianPDF_batchmean(Zk[j],Zmk,R)
                wmc[yy] = TP*likez*wmc[yy]
                
                Zmk = robots[j].sensormodel.evalBatchNoFOV(t, dt, Xmc[~yy][:,target.posstates],posstates=target.posstates)
                likez = uqstpdf.gaussianPDF_batchmean(Zk[j],Zmk,R)
                wmc[~yy] = FP*likez*wmc[~yy]
            else:
                wmc[yy] = FN*wmc[yy]
                wmc[~yy] = TN*wmc[~yy]
            
            # Xmc = np.vstack([Xmcin,Xmcout])
            # wmc = np.hstack([win,wout])
                                   
            wmc = wmc/np.sum(wmc)
    isresampled = False
    if np.allclose(wmc, wmc_old, rtol=1e-07, atol=1e-09):
        Xmc_resample = Xmc
    else:
        isresampled = True
        with utltm.TimingContext("resample: ",printit = False):
            I = myresampling(wmc)
            I = np.round(I).astype(int)
            Xmc_resample = np.zeros(Xmc.shape)
            Xmc_resample = Xmc[I,:]+0.001*np.random.randn(Xmc.shape[0],Xmc.shape[1])
        
    if computePriorPostDist == True:
        with utltm.TimingContext("Hist dist: ",printit = False):
        
            H1, edges = np.histogramdd(Xmc[:,target.posstates], bins=infoconfig.bins[0:2])
            H2, _ = np.histogramdd(Xmc_resample[:,target.posstates], bins=infoconfig.bins[0:2])
            edges = np.array(edges)
            aa=np.diff(edges,axis=1)
            area= np.prod(aa[:,0])
            vol1 = np.sum(H1.reshape(-1)*area)
            vol2 = np.sum(H2.reshape(-1)*area)
            H1=H1/vol1
            H2=H2/vol2
            # print(H1)
            dd = np.linalg.norm(H1-H2)
            # dd = 1e1*(-np.sum(H1*np.log(H1+1)) +  np.sum(H2*np.log(H2+1)))
            if t not in target.context:
                target.context[t]={'info': dd}    
            else:
                target.context[t]['info'] = dd    
        
    
    
    if isresampled and mergerConfig.doMerge:
        with utltm.TimingContext("GMM fit ",printit = False):
            clf = mixture.GaussianMixture(n_components=mergerConfig.fixedComps, covariance_type='full')
            clf.fit(Xmc_resample)
            gmmuk = uqgmmbase.GMM(None,None,None,0)
            # gmmuk.appendCompArray(clf.means_,np.stack([clf.covariances_]*mergerConfig.fixedComps,axis=0),clf.weights_,renormalize = True)
            gmmuk.appendCompArray(clf.means_,clf.covariances_,clf.weights_,renormalize = True)
            gmmuk.pruneByWts(wtthresh=mergerConfig.wtthreshprune,renormalize = True)    
    else:
        gmmuk = target.gmmfk
            
            
    # Nmc = 1000
    # Xmc1 = gmmuk.random(Nmc)
    # wmc = np.ones(Nmc)/Nmc
    # gmmukcollap = gmmuk.collapseGMM()
    # Xmcollap = gmmukcollap.random(Nmc)
    # XX = np.vstack([Xmc1,Xmcollap])
    # p1 = gmmuk.pdf(XX)
    # p2 = gmmukcollap.pdf(XX)
    # mm = np.minimum(p1,p2)
    # yy = mm>1e-6
    # dd = np.abs(p1-p2)
    # f = 100*dd[yy]/mm[yy]
    
    # if np.mean(f)<75:
    #     gmmuk = gmmukcollap
    #     print("collapsed gmm")

    
    # print("-------UPDATE END--------")
    xfk,Pfk = gmmuk.meanCov()
    if updttarget:
        target.setTargetFilterStageAsPosterior()
        target.updateParams(currt=t,gmmfk=gmmuk,xfk=xfk,Pfk=Pfk)
        
    
    return gmmuk
    
            
            