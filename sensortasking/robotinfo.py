import numpy as np
import numpy.linalg as nplnalg
import scipy.linalg as sclnalg
from uq.information import distance as uqinfodis
import robot.filters.robot2Dfilters as rbf2df
import collections as clc
from uq.gmm import gmmbase as uqgmmbase
import utils.timers as utltm
from random import shuffle
import multiprocessing as mp
import matplotlib.pyplot as plt
from utils.math import geometry as utmthgeom
import pickle as pkl

def plotinfomap(simmanager,targetset,robots,infocost,ridstates,t):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure("Info-Surf: %f"%t)
    print("working")
    colmap = plt.get_cmap('gist_rainbow')
    NUM_COLORS = 20
    colors = [colmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    shuffle(colors)

    
    ax_list = fig.axes
    if len(ax_list)==0:     
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax_list[0]
    ax.cla()
    
    for i in range(targetset.ntargs):
        # xktruth = targetset[i].groundtruthrecorder.getvar_uptotime_stacked('xtk',t)
        gmmu = targetset[i].recorderprior.getvar_bytime('gmmfk',t)
        # if gmmu is None:
        #     gmmu = targetset[i].recorderprior.getvar_bytime('gmmfk',t+dt)
            
        gmmupos = uqgmmbase.marginalizeGMM(gmmu,targetset[i].posstates)
        XX = uqgmmbase.plotGMM2Dcontour(gmmupos,nsig=2,N=100,rettype='list')
        for cc in range(gmmupos.Ncomp):
            ax.plot(XX[cc][:,0],XX[cc][:,1],c=colors[i])
            # ax.annotate("wt: "+str(gmmupos.w(cc))[:5],gmmupos.m(cc)[0:2],gmmupos.m(cc)[0:2]-3,color=colors[i],fontsize='x-small')
        # ax.plot(xktruth[:,0],xktruth[:,1],linestyle='--',c=colors[i])
        # ax.plot(xktruth[-1,0],xktruth[-1,1],c=colors[i],marker='*')
        
        
    robots[0].mapobj.plotmap(ax)
    # for r in robots:
    #     r.plotrobot(ax)
            
    p=np.zeros(ridstates.shape[0])
    x=np.zeros(ridstates.shape[0])
    y=np.zeros(ridstates.shape[0])
    for n in range(ridstates.shape[0]):
        x[n] = ridstates[n,0]
        y[n] = ridstates[n,1]
        p[n] = infocost[t][n] 
    ax.scatter(x, y, p, c='b',marker='o')
    plt.pause(1)
    plt.ion()
    # plt.show(block=True)
    plt.show()
    plt.pause(0.1)
    k = f'{t:06.2f}'.replace('.','-')
    simmanager.savefigure(fig, ['SimSnapshot', 'InfoMap'], k+'.png',data=[targetset,robots,infocost,ridstates,t])

def getinfotimestep(j,rid,ridstates,robots,targetset,tvec,Targetfilterer,infoconfig,splitterConfig,mergerConfig):
    D={}
    for n in range(ridstates.shape[0]):
        # with utltm.TimingContext("robot pos: "):
        
        robots[rid].xk[0:2] = ridstates[n] 
        robots[rid].updateSensorModel()
        for r in range(rid):
            robots[r].xk = robots[r].statehistory[tvec[j]]
            robots[r].updateSensorModel()
        
        It=0    
        for i in range(targetset.ntargs):
            dt = tvec[j]-tvec[j-1]
            # gmmfk = targetset[i].recorderprior.getvar_bytime('gmmfk',tvec[j])
            targetset[i].setStateFromPrior(tvec[j],['gmmfk','xfk','Pfk'])
            
            gmmfkpos = uqgmmbase.marginalizeGMM(targetset[i].gmmfk,targetset[i].posstates)
            Nz = 1000
            X = gmmfkpos.random(Nz)
            w = np.ones(Nz)/Nz
            # X,w = gmmfkpos.generateUTpts()

            
            I=[]
            cache={}
            for s in range(X.shape[0]):
                Zk=clc.defaultdict(list)
                for r in range(rid+1):
                    zij,isinsidek,Lk = robots[r].sensormodel.generateRndMeas(tvec[j], dt, X[s],useRecord=False)     
                    if isinsidek == 0:
                        Zk[i].append(None)
                    else:
                        Zk[i].append(zij)
                
                # measUpdateFOV_PFGMM
                # measUpdateFOV_randomsplitter
                ykey = tuple([1 if zz is not None else 0 for zz in Zk[i]])
                if ykey not in cache:
                    dm = mergerConfig.doMerge
                    mergerConfig.doMerge = False
                    gmmuk = rbf2df.measUpdateFOV_randomsplitter(tvec[j],dt,robots[:(rid+1)],targetset[i],Zk[i],Targetfilterer,infoconfig,
                                                    updttarget=False,
                                                    splitterConfig = splitterConfig,
                                                    mergerConfig = mergerConfig,
                                                    computePriorPostDist=True)
                    mergerConfig.doMerge = dm
                    

                    dd = targetset[i].context[tvec[j]]['info']
                    cache[ykey] = dd
                else:
                    dd = cache[ykey]
                    
                # I=I+w[s]*dd
                I.append(dd)
            
            I = np.min(I)
            It = It + I
        
        print("Done info grid for %d/%d at timestep %d/%d"%(n,ridstates.shape[0],j,len(tvec)))
        D[n] = It

    return D

    
def robotTargetInfo(simmanager,rid,ridstates,robots,targetset,tvec,Targetfilterer,infoconfig,splitterConfig,mergerConfig):
    # with open("debug-info-tvec-"+str(int(tvec[0]))+'.pkl','wb') as FF:
    #     pkl.dump([rid,ridstates,robots,targetset,tvec,Targetfilterer,splitterConfig,mergerConfig],FF)
        
    for i in range(targetset.ntargs):
        targetset[i].freezeState()
    for r in range(rid+1):
        robots[r].freezeState()    

    D=clc.defaultdict(dict)
    for n in range(ridstates.shape[0]):
        D[tvec[0]][n]=0
        
    for j in range(1,len(tvec)):
        for n in range(ridstates.shape[0]):
            # with utltm.TimingContext("robot pos: "):
            It=0
            robots[rid].xk[0:2] = ridstates[n] 
            robots[rid].updateSensorModel()
            for r in range(rid):
                robots[r].xk = robots[r].statehistory[tvec[j]]
                robots[r].updateSensorModel()
                
            for i in range(targetset.ntargs):
                dt = tvec[j]-tvec[j-1]
                # gmmfk = targetset[i].recorderprior.getvar_bytime('gmmfk',tvec[j])
                targetset[i].setStateFromPrior(tvec[j],['gmmfk','xfk','Pfk'])
                
                gmmfkpos = uqgmmbase.marginalizeGMM(targetset[i].gmmfk,targetset[i].posstates)
                Nz = 1000
                X = gmmfkpos.random(Nz)
                w = np.ones(Nz)/Nz
                # X,w = gmmfkpos.generateUTpts()

                
                I=[]
                cache={}
                for s in range(X.shape[0]):
                    Zk=clc.defaultdict(list)
                    for r in range(rid+1):
                        zij,isinsidek,Lk = robots[r].sensormodel.generateRndMeas(tvec[j], dt, X[s],useRecord=False)     
                        if isinsidek == 0:
                            Zk[i].append(None)
                        else:
                            Zk[i].append(zij)
                    
                    # measUpdateFOV_PFGMM
                    # measUpdateFOV_randomsplitter
                    ykey = tuple([1 if zz is not None else 0 for zz in Zk[i]])
                    if ykey not in cache:
                        dm = mergerConfig.doMerge
                        mergerConfig.doMerge = False
                        gmmuk = rbf2df.measUpdateFOV_randomsplitter(tvec[j],dt,robots[:(rid+1)],targetset[i],Zk[i],Targetfilterer,infoconfig,
                                                        updttarget=False,
                                                        splitterConfig = splitterConfig,
                                                        mergerConfig = mergerConfig,
                                                        computePriorPostDist=True)
                        mergerConfig.doMerge = dm
                        

                        dd = targetset[i].context[tvec[j]]['info']
                        cache[ykey] = dd
                    else:
                        dd = cache[ykey]
                        
                    # I=I+w[s]*dd
                    I.append(dd)
                
                I = np.min(I)
                It = It + I
            
            print("Done info grid for %d/%d at timestep %d/%d"%(n,ridstates.shape[0],j,len(tvec)))
            D[tvec[j]][n] = It
            
        plotinfomap(simmanager,targetset,robots[:(rid+1)],D,ridstates,tvec[j])
        
    for i in range(targetset.ntargs):
        targetset[i].defrostState()
    for r in range(rid+1):
        robots[r].defrostState()  
        robots[r].updateSensorModel()
        
    return D