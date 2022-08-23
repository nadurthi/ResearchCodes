import scipy
import math
import numpy as np
from numpy import linalg as nplinalg
from scipy import linalg as sclinalg
import matplotlib.pyplot as plt
import pandas as pd
import ssatoolkit as ssatl
import ssatoolkit.constants as cst
#import laplace_method as iod
from ssatoolkit import mht
from ssatoolkit import coords
from ssatoolkit import filters
from ssatoolkit.iod import laplace_method as lpmthd
from ssatoolkit.iod import doubler_iteration as dbriter
from ssatoolkit import ukf
import ssatoolkit.propagators as ssaprop
from ssatoolkit.sensors import Sensor, SensorSet
from ssatoolkit.targets import TrueTargets, CatalogTargets
import os
import plotting
import networkx as nx
from sklearn.cluster import KMeans
from ssatoolkit.od import differential_correction
from ssatoolkit import propagators
from ssatoolkit import filters
# all simulations are with respect to fixed Earth
# so coordinate system is fixed
import pdb
import time
import simmanager
import copy
import scipy

import time
import logging
import progressbar

progressbar.streams.wrap_stderr()
logging.basicConfig()


    
plt.close('all')


t0 = 0
tf = 3*24 * 3600  #
dt = 1*60  # seconds
Tvec = np.arange(t0, tf, dt)

# %% targets
pklfile = 'data/randomOrbits.pkl'

# Data = TrueTargets(os.path.join('data','100_brightest.csv'),cst.Earth.mu)

Data = TrueTargets(cst.Earth.mu)
t0, dt, tf, Tvec = Data.get_pickle(pklfile)

Ntargs = 10
Data.setMaxTargets(Ntargs)

Data.generate_trajectory(Tvec)


# Data.getMeasStats(sensors,t0, tf, dt,NminMeas = 35)

# %%


try:
    runfilename = __file__
except:
    runfilename = "main_run.py"

metalog = """
AAS Paper 2022
SSA, MHT, Sensor Tasking

"""

simmanger = simmanager.SimManager(t0=t0, tf=tf, dt=dt, dtplot=dt/10,
                                  simname="SSA-MHT-SENSOR-TASKING-%d_Targs" % (Ntargs,), savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()
simmanger.data['Ntargs'] = Ntargs


# %% sensors

sensors = SensorSet()
sensors.append(Sensor(idx=1, Rk=np.diag([(0.000001*np.pi/180)**2, (0.000001*np.pi/180)**2]),
                      earthpos=[-23.02331*coords.to_rad(), -67.75379 *
                                coords.to_rad(), cst.Earth.radius],
                      half_angle=60*coords.to_rad(), rmax=5000, orientation='normal'))
sensors.append(Sensor(idx=2, Rk=np.diag([(0.000001*np.pi/180)**2, (0.000001*np.pi/180)**2]),
                      earthpos=[49.133949*coords.to_rad(), 1.441930 *
                                coords.to_rad(), cst.Earth.radius],
                      half_angle=60*coords.to_rad(), rmax=5000, orientation='normal'))
sensors.append(Sensor(idx=3, Rk=np.diag([(0.000001*np.pi/180)**2, (0.000001*np.pi/180)**2]),
                      earthpos=[35.281287*coords.to_rad(), -116.783051 *
                                coords.to_rad(), cst.Earth.radius],
                      half_angle=60*coords.to_rad(), rmax=5000, orientation='normal'))


    

def plotsim(k0, k, sensors, Data, estTargets, planet, savefig=False):

    figsim = plt.figure("FullSim", figsize=(20, 10))
    if len(figsim.axes) == 0:
        ax = figsim.add_subplot(111, projection='3d')
    else:
        ax = figsim.axes[0]
        ax.cla()

    # plot earth
    plotting.plotPlanet(ax, planet)

    # plot traj
    for key in Data.targIds:
        #print(true_trajectories[key][:, 0])
        ax.plot3D(Data.true_trajectory[key][k0:k, 0],
                  Data.true_trajectory[key][k0:k, 1], Data.true_trajectory[key][k0:k, 2], 'black')

    # plot sensors
    for sens in sensors.itersensors():
        plotting.plot_cone(ax, sens.r_site, sens.orientation,
                           sens.rmax, sens.half_angle, sens.plotparams)

    # plot estimated target tracks
    if estTargets is not None:
        for esttraj in estTargets:
            #print(true_trajectories[key][:, 0])
            ax.plot3D(esttraj[:, 0], esttraj[:, 1], esttraj[:, 2], 'r--')

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('z', fontsize=20)

    if savefig:
        simmanger.savefigure(figsim, [
                             '3Dsim', 'snapshot'], 'snapshot_'+str(int(k)), data=[k0, k, estTargets, planet])


# %% targets


estTargets = None
plotsim(0, len(Tvec), sensors, Data, estTargets, cst.Earth, savefig=True)


# %%


thres_sigma_sqrd = 3
mhtree = mht.MHT(thres_sigma_sqrd)
simmanger.data['mhtree_history'] = {0: mhtree}
simmanger.data['globalOrbs_history'] = {}
simmanger.data['mhtree_history_before_isolation']={0:mhtree}
simmanger.data['mhtree_history_after_isolation']={0:mhtree}
# nx.draw(mhtree.measGraph)
# G = mhtree.measGraph
# nx.draw_networkx_labels(G, pos=nx.spring_layout(G))


mhtfilters = {}
mhtfilters['cov_sigma_filter'] = {'enforce': True, 'thres_sigma_sqrd': 3**2}
mhtfilters['planar_filter'] = {
    'enforce': False, 'steps': 10, 'half_angle_thresh': 10*coords.to_rad()}
mhtfilters['iod_filter'] = {'enforce': False, 'orbRandIter': 5, 'angleThres': 20*coords.to_rad(), 'VoteFrac': 0.5,
                            'minBranchLenOrbs_Inactive': 10, 'OB_lastnode_min_voters': 10}


Qk = np.diag([0.005**2, 0.005**2, 0.005**2, 0.01**2, 0.01**2, 0.01**2])

simmanger.data['true_orbs'] = true_orbs = Data.getOrbs()
figs = {}
mainbreak=False
# main loop of the simulation
for k in range(1, len(Tvec)):  #

    # this function propagates true targets to timestep k,
    M = Data.get_measurements(sensors, k)
    print("k= ", k, "#Meas = ", len(M), " Active,InActive: ", len(
        mhtree.getActiveLeafNodes()), len(mhtree.getInActiveLeafNodes()))
    # mhtree.getEstimateMetrics(Data,Tvec,cst.Earth,clusterWidths={'a':100,'e':0.1,'i':10*np.pi/180,'Om':10*np.pi/180,'om':10*np.pi/180,'f':20*np.pi/180})

    # checks if they are within FOV, generates measurements with noise
    # print(M)

    mhtree.add_measurements(k, Tvec, M, cst.Earth.mu)

   
    
    # now do UKF stuff
    if mhtfilters['cov_sigma_filter']['enforce'] is True:
        branches = mhtree.iterate_last_K_step_active_branches(K=-1)
        for measbranch in branches:
            MM = mhtree.getMeasFromNodes(measbranch)

            # if last measurement is None, skip branch
            # if MM[-1] is None:
            #     continue
            lastnode = measbranch[-1]
            if mhtree.measGraph.nodes[lastnode].get('meas_idx',None) is None:
                continue
            
            tklast = mhtree.measGraph.nodes[lastnode]['timek']
            tlast = Tvec[tklast]

            idx_mk_recent = None
            for i in range(len(measbranch)-2, 0, -1):
                node = measbranch[i]
                if mhtree.measGraph.nodes[node].get('od_rv', None) is not None:
                    idx_mk_recent = i
                    break

            if idx_mk_recent is None:
                continue
            node = measbranch[idx_mk_recent]
            kp = mhtree.measGraph.nodes[node]['timek']
            mp = mhtree.measGraph.nodes[node]['od_rv']
            Pp = mhtree.measGraph.nodes[node].get('od_Prv',None)
            if Pp is None:
                Pp = np.diag([10**2,10**2,10**2,1**2,1**2,1**2])
                
            mk, Pk = ukf.propagateUKF(
                Tvec[kp], Tvec[k], mp, Pp, ssaprop.FnG, Qk, cst.Earth.mu)  # k,kp,mp,Pp

            klast = mhtree.measGraph.nodes[lastnode]['timek']
            meas_idx = mhtree.measGraph.nodes[lastnode]['meas_idx']
            sensID = mhtree.measGraph.nodes[lastnode]['sensID']

            zk = mhtree.history_M[klast][meas_idx].zk
            sens = sensors[sensID]

            # get measurement likelihood
            xu, Pu, mz, Pz = ukf.measUpdateUKF(mk, Pk, sens.evalfunc, zk, sens.Rkpos)
            lgpdf = np.matmul(np.matmul(zk-mz, nplinalg.inv(Pz)), zk-mz)
            
            if lgpdf > mhtfilters['cov_sigma_filter']['thres_sigma_sqrd']:
                # meas zk failed to be within predicted covariance of the target
                # mhtree.measGraph.nodes[lastnode]['status'] = 'InActive'
                mhtree.makeNodeInActive(lastnode,['UKF filter rejected',lgpdf,Pp])
                
                
    # now do IOD stuff
    st = time.time()
    branches = mhtree.iterate_last_K_step_active_branches(K=-1)
    for measbranch in branches:
        lastnode = measbranch[-1]
        tklast = mhtree.measGraph.nodes[lastnode]['timek']
        tlast = Tvec[tklast]

        MM = mhtree.getMeasFromNodes(measbranch)
        MMn = [mm for mm in MM if mm is not None]

        
        for sens in sensors.itersensors():
            # all measurements have to belong to same sensor
            MMs = [mm for mm in MMn if mm.sensID == sens.idx]
            
            if len(MMs) <= 4 :
                continue
            
            if MMs[-1].k!=tklast:
                # brnach has to end in a node that is not None
                continue
            
            

            Nm = len(MMs)

            DONE = []
            for i in range(1, 5):
                i2 = Nm-1-i
                i1 = Nm-1-2*i
                if i1 > 0 and i2 >= 0 and i1 < i2 and i2 != Nm-1 and (i1, i2) not in DONE:
                    DONE.append((i1, i2))
                    T = [Tvec[MMs[i1].k], Tvec[MMs[i2].k], Tvec[MMs[-1].k]]
                    
                    L = [MMs[i1].zk, MMs[i2].zk, MMs[-1].zk]
                    Tchecks = [Tvec[MMs[i1-1].k]]
                    Lchecks = [MMs[i1-1].zk]
                    if np.max(np.abs(np.diff(T[-3:]))) > 5*dt:
                        continue
                        

                    rvdbr = lpmthd.laplace_orbit_fit(
                        sens.r_site, T, L, Tchecks, Lchecks, cst.Earth.mu, 20*coords.to_rad(), 0.25)

                    nodei2 = MMs[i2].nodeID
                    if len(rvdbr) > 0:
                        RVS = []
                        for j in range(len(rvdbr)):
                            X = ssaprop.propagate_rv(
                                T[1], rvdbr[j][:3], rvdbr[j][3:], tlast, cst.Earth.mu)
                            RVS.append(X[-1])
                        # rvlast = X[-1]
                        if 'iod_rv' not in mhtree.measGraph.nodes[lastnode].keys():
                            mhtree.measGraph.nodes[lastnode]['iod_rv'] = RVS
                        else:
                            mhtree.measGraph.nodes[lastnode]['iod_rv'].extend(RVS)     # get all the rv propagated from the prevnode

        for i in range(len(measbranch)-2, -1, -1):
            prevNode = measbranch[i]
            tprev = Tvec[mhtree.measGraph.nodes[prevNode]['timek']]
            if 'iod_rv' not in mhtree.measGraph.nodes[prevNode].keys():
                continue
            RVs = mhtree.measGraph.nodes[prevNode]['iod_rv']
            # print("len(RVs) = ",len(RVs))
            for rv in RVs:
                X = ssaprop.propagate_rv(
                    tprev, rv[:3], rv[3:], tlast, cst.Earth.mu)
                # X=ssaprop.propagate_FnG(np.array([tprev,tlast]), rv, cst.Earth.mu, tol = 1e-12)
                rvlast = X[-1]
                if 'iod_rv' not in mhtree.measGraph.nodes[lastnode].keys():
                    mhtree.measGraph.nodes[lastnode]['iod_rv'] = [rvlast]
                else:
                    mhtree.measGraph.nodes[lastnode]['iod_rv'].append(rvlast)
            break

        # compute the scores
        data = differential_correction.M2data(MMn, sensors, Tvec)
        RVslast = mhtree.measGraph.nodes[lastnode].get('iod_rv', None)
        tlast = Tvec[mhtree.measGraph.nodes[lastnode]['timek']]
        if RVslast is None:
            mhtree.measGraph.nodes[lastnode]['best_iod_score'] = None
            mhtree.measGraph.nodes[lastnode]['best_iod_rv'] = None
        else:
            scores = []
            for i in range(len(RVslast)):
                rvlast = RVslast[i]
                errs = differential_correction.zkDistErrs(
                    rvlast, tlast, data, cst.Earth.mu)
                scores.append(np.max(errs))

            mhtree.measGraph.nodes[lastnode]['iod_scores'] = scores
            mhtree.measGraph.nodes[lastnode]['best_iod_score'] = np.min(scores)
            ix = np.argmin(scores)
            mhtree.measGraph.nodes[lastnode]['best_iod_rv'] = RVslast[ix]
            # mhtree.measGraph.nodes[lastnode]['rv']=[RVslast[ix]]
            # mhtree.measGraph.nodes[lastnode]['scores']=[np.min(scores)]

    et = time.time()
    print("IOD stuff time taken = ", et-st)

    # now do od assignment -------------------------------------------
    computeOD = True
    if computeOD:
        leafNodes = mhtree.getActiveLeafNodes()
        leafScores = []
        # if len(leafNodes) > 50:
        for lnode in leafNodes:
            # if mhtree.measGraph.nodes[lnode]['meas_idx'] is None:
            #     continue
            best_score = mhtree.measGraph.nodes[lnode].get('best_iod_score', None)
            if best_score is not None:
                leafScores.append((best_score, lnode))

                    
        leafScores = sorted(leafScores, key=lambda x: x[0])
        for i in progressbar.progressbar(range(len(leafScores))):

            # st = time.time()
            lnode = leafScores[i][1]
            score = leafScores[i][0]

            if lnode not in mhtree.measGraph.nodes:
                continue
            
            if mhtree.measGraph.nodes[lnode]['meas_idx'] is not None and mhtree.measGraph.nodes[lnode]['best_iod_score']>0.5:
                # print("candidate for OD-----: ", lnode)
                rv = mhtree.measGraph.nodes[lnode]['best_iod_rv']
                tk = mhtree.measGraph.nodes[lnode]['timek']
                T = Tvec[tk]
                orb = coords.from_RV_to_OE(rv[0:3], rv[3:], cst.Earth.mu)
                brch = mhtree.getCompleteBranch(lnode)
                MM = mhtree.getMeasFromNodes(brch)
                MM = [m for m in MM if m is not None]
                # if len(MM)<10:
                #     continue
                data = differential_correction.M2data(MM, sensors, Tvec)
    
                if data.shape[0] < 6:
                    continue
    
                st1 = time.time()
                res = differential_correction.orbit_det(
                    T, rv, data, cst.Earth)
                rv = res['x']
                et1 = time.time()
                # print("diff_corr = ", et1-st1)
    
                errs = differential_correction.zkDistErrs(
                    res['x'], T, data, cst.Earth.mu)
                score = np.max(errs)
            else:
                rv = mhtree.measGraph.nodes[lnode]['best_iod_rv']
                score = mhtree.measGraph.nodes[lnode]['best_iod_score']
                
            # if score < 0.5:
                # print("Yay.... isolating a branch after OD confirmation")

                # mhtree.isolate_branch(lnode, makeInActive=False)
                
                
            mhtree.measGraph.nodes[lnode]['od_rv'] = rv
            mhtree.measGraph.nodes[lnode]['od_score'] = score
            mhtree.measGraph.nodes[lnode]['iod_scores'] = [score]
            mhtree.measGraph.nodes[lnode]['iod_rv'] = [rv]
            mhtree.measGraph.nodes[lnode]['best_iod_score'] = score
            mhtree.measGraph.nodes[lnode]['best_iod_rv'] = rv

            # else:
            #     if lnode in mhtree.measGraph.nodes:
            #         mhtree.measGraph.nodes[lnode]['status'] = 'InActive'

                    
                
                # et = time.time()
                # print("greedy time = ", et-st)
                
    # brute force remove nodes
    leafNodes = mhtree.getActiveLeafNodes()
    for lnode in leafNodes:
        rv = mhtree.measGraph.nodes[lnode].get('od_rv',None)
        score = mhtree.measGraph.nodes[lnode].get('od_score',None)
        if score is not None and score > 0.5:
            mhtree.measGraph.nodes[lnode]['status'] = 'InActive'
            
        if rv is not None:
            orb = coords.from_RV_to_OE(
                rv[:3], rv[3:], cst.Earth.mu)
            a, e, ii, Om, om, f = orb
            if (a*(1-e) < cst.Earth.radius or e < 0 or e > 1 or a < 0 or a > 36000):
                mhtree.makeNodeInActive(lnode,['bad orb elments',orb])
            
        # no OD computed even after multiple measurements
        brch = mhtree.getCompleteBranch(lnode)
        N=len(brch)
        MM = mhtree.getMeasFromNodes(brch)
        iodcnt=0
        odcnt=0
        meascnt=0
        for i in range(len(brch)-1,-1,-1):
            if mhtree.measGraph.nodes[brch[i]].get('iod_rv',None) is not None:
                iodcnt+=1
            if mhtree.measGraph.nodes[brch[i]].get('od_rv',None) is not None:
                odcnt+=1
            if mhtree.measGraph.nodes[brch[i]].get('meas_idx',None) is not None:
                meascnt+=1
            
        if meascnt>=7 and iodcnt==0:
            mhtree.measGraph.nodes[lnode]['status'] = 'InActive'
    
        if meascnt>=7 and odcnt==0:
            mhtree.measGraph.nodes[lnode]['status'] = 'InActive'
            
                    
        
                    
       


    et = time.time()
    # simmanger.data['mhtree_history_before_isolation'][k] = copy.deepcopy(mhtree)
    
    mhtree_saved = copy.deepcopy(mhtree)
    DoGLobalIsolation  = True
    leafNodes = mhtree.getActiveLeafNodes()
    print("--------------before isolation : %d--------------------"%k)
    dfbefore = mhtree.getEstimateMetrics(Data,Tvec,cst.Earth,clusterWidths={'a':100,'e':0.1,'i':10*np.pi/180,'Om':10*np.pi/180,'om':10*np.pi/180,'f':20*np.pi/180})
    if DoGLobalIsolation and len(leafNodes)>1000:
        leafScores=[]
        for lnode in leafNodes:
            best_score = mhtree.measGraph.nodes[lnode].get('od_score', None)
            if best_score is not None:
                leafScores.append((best_score, lnode))
                # print((best_score, lnode))
    
        leafScores = sorted(leafScores, key=lambda x: x[0])
        
        
        for i in range(len(leafScores)):
    
            # st = time.time()
            lnode = leafScores[i][1]
            if lnode not in mhtree.measGraph.nodes:
                continue
            if mhtree.measGraph.nodes[lnode]['status'] =='InActive' :
                continue
            
            score = leafScores[i][0]
            cutoff_good_score =0.5
            if score < cutoff_good_score:
                print("Yay.... isolating a branch after OD confirmation")
                overlapnodes = mhtree.getOverlappedBranches(lnode)
                # overlapnodes = [nn for nn in overlapnodes if mhtree.measGraph.nodes[nn]['od_score']<cutoff_good_score]
                globalOrbs = mhtree.getActiveOrbs(Tvec,cst.Earth,leafNodes=overlapnodes)

                
                mhtree.isolate_branch(lnode, makeInActive=False)
    
    
    # global clustering
    globalOrbs = mhtree.getActiveOrbs(Tvec,cst.Earth)
    print("------------------")
    print("len(globalOrbs) = ", len(globalOrbs))
    print("------------------")
    
    # simmanger.data['globalOrbs_history'][k] = globalOrbs
    DoClusterFusion=False
    leafNodes = mhtree.getActiveLeafNodes()
    if len(globalOrbs)>20 and DoClusterFusion:

        cluster_orbs=filters.getOrbClusters(globalOrbs,clusterWidths={'a':100,'e':0.1,'i':10*np.pi/180,'Om':10*np.pi/180,'om':10*np.pi/180,'f':20*np.pi/180})
        
        for i in range(len(cluster_orbs)):
            pts = cluster_orbs[i]
            if len(pts)==1:
                continue
            Lorbcls = [tuple(row) for row in pts[:,-2:].astype(int)]
            Lorbcls = sorted(Lorbcls,key=lambda x: mhtree.measGraph.nodes[x]['od_score'])
            Lorbcls_act=[]
            Nmeas=[]
            for j in range(len(Lorbcls)):
                # brmain = mhtree.getCompleteBranch(Lorbcls[j])
                if mhtree.measGraph.nodes[Lorbcls[j]]['od_score']<0.5:
                    mhtree.isolate_branch(Lorbcls[j],makeInActive=False)
                    Lorbcls_act.append(Lorbcls[j])
                    brmain = mhtree.getCompleteBranch(Lorbcls[j])
                    MM = mhtree.getMeasFromNodes(brmain)
                    MM = [m for m in MM if m is not None]
                    Nmeas.append(len(MM))        
                else:
                    mhtree.makeNodeInActive(Lorbcls[j], ['od_Score based elimination'])
                    
    
            
            # #next pick the one with lot of measurements from the set of good branches
            if len(Nmeas)>0:
                Nmeas=np.array(Nmeas)
                idx = np.argmax(Nmeas)
                best_node = Lorbcls_act[idx]
                for j in range(len(Lorbcls_act)): 
                    if best_node==Lorbcls_act[j]:
                        continue
                    mhtree.makeNodeInActive(Lorbcls_act[j], ['best nopde was: ',best_node,' hence inactivating this branch'])
                
    print("--------------after isolation : %d--------------------"%k)
    dfafter = mhtree.getEstimateMetrics(Data,Tvec,cst.Earth,clusterWidths={'a':100,'e':0.1,'i':10*np.pi/180,'Om':10*np.pi/180,'om':10*np.pi/180,'f':20*np.pi/180})
    # simmanger.data['mhtree_history_after_isolation'][k] = copy.deepcopy(mhtree)
    
    # if any(dfafter.sort_values(by='Target')['Nestorbs'].values-dfbefore.sort_values(by='Target')['Nestorbs'].values <0):
    #     break
    
    DoOrbEstPlot= True
    if DoOrbEstPlot:
        globalOrbs = mhtree.getActiveOrbs(Tvec,cst.Earth)  
        if len(globalOrbs) > 0:
            orbstates = ['a', 'e', 'i', 'Om', 'om', 'f']
            for pp in [0, 1, 2, 3]:
                figs[pp] = plt.figure("%s-%s" % (orbstates[pp], orbstates[pp+1]))
                if len(figs[pp].axes)==0:
                    ax = figs[pp].add_subplot(111)
                else:
                    ax = figs[pp].axes[0]
                    ax.cla()
                ax.plot(globalOrbs[:, pp], globalOrbs[:, pp+1], 'ro', markersize=12)
                ax.plot(true_orbs[:, pp], true_orbs[:, pp+1], 'k*', markersize=10)
                ax.set_xlabel(orbstates[pp], fontsize=20)
                ax.set_ylabel(orbstates[pp+1], fontsize=20)
                ax.grid()
                ax.tick_params(axis='both', which='major', labelsize=20)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                ax.set_title("time step = %d"%k)
                # if pp==0:
                #     simmanger.savefigure(figs[pp], ['Orb2Dest'], 'orbEst_%d_%s_%s_'%(int(k),orbstates[pp],orbstates[pp+1]),data=[globalOrbs])
                # else:
                #     simmanger.savefigure(figs[pp], ['Orb2Dest'], 'orbEst_%d_%s_%s_'%(int(k),orbstates[pp],orbstates[pp+1]),data=[])
        
                plt.show()
                plt.pause(0.1)
                


    estTargets = None
    leafNodes = mhtree.getActiveLeafNodes()
    for lnode in leafNodes:
        rvk = mhtree.measGraph.nodes[lnode].get('best_rv', None)
        if rvk is None:
            continue
        # orb = coords.from_RV_to_OE(rvk[0:3],rvk[3:],cst.Earth.mu)
        # a,e,i,Om,om,f = orb

        tk = Tvec[mhtree.measGraph.nodes[lnode]['timek']]
        _, rv_traj = ssaprop.propagate_rv_tvec(
            tk, rvk, Tvec[0:k], cst.Earth.mu)
        if estTargets is None:
            estTargets = []
        if np.any(np.isnan(rv_traj)):
            print(lnode)
            mhtree.makeNodeInActive(lnode, ['traj has nans'])
            # mhtree.measGraph.nodes[lnode]['status'] = 'InActive'
            # mhtree.measGraph.nodes[lnode]['status_reason'] = ['traj has nans']
        else:
            estTargets.append(rv_traj)
    # plotsim(0,k,sensors,Data,estTargets,cst.Earth,savefig=True)

# %%
simmanger.finalize()

simmanger.save(metalog, mainfile=runfilename, sensors=sensors,
               Data=Data, Tvec=Tvec, mhtfilters=mhtfilters, mhtree=mhtree)
