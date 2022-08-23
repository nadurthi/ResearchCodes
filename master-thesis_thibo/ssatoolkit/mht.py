# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import uuid
import pdb
from numpy import linalg as nplinalg
from ssatoolkit import coords
import numpy as np
import scipy.sparse as sp
import ssatoolkit.targets as ssatarg
import copy
import collections as clc
import pandas as pd
try:
    import gurobipy as gp
    from gurobipy import GRB
except:
    print("No Gurobi")

# def getSensorTensorProd(M,sensors):
#     # M=[(s1,'a',L,k),(s2,'ra',r,k),..]
#     # get tensor product of sensor measurements 
#     fovM = np.zeros((len(M),len(sensors)),dtype=bool)
    
    
#     # Msens={}
#     # for m in M:
#     #     if m[0] not in Msens.keys():
#     #         Msens[m[0]]=[]
#     #     Msens[m[0]].append(m)
    
#     # take tensor product of the ones that are common
#     for j in range(len(M)):
#         for i,sens in enumerate(sensors.itersensors()):
#             fovM[j,i] =  sens.inFOV()               
        
        
class MHT:
    def __init__(self,thres_sigma_sqrd):
        self.history_M={}
        self.measGraph = nx.DiGraph()
        self.cnt=0
    
    def getLeafNodes(self):
        leafnodes = [x for x in self.measGraph.nodes() if self.measGraph.out_degree(x)==0 ] #and self.measGraph.in_degree(x)==1
        
        return leafnodes
    def getrootnodes(self):
        rootnodes = [x for x in self.measGraph.nodes() if self.measGraph.in_degree(x)==0 ] #and self.measGraph.in_degree(x)==1
        return rootnodes
        
    def getActiveLeafNodes(self):
        activeleafnodes = [x for x in self.measGraph.nodes() if self.measGraph.out_degree(x)==0 and self.measGraph.nodes[x]['status']=='Active' ] #and self.measGraph.in_degree(x)==1
        
        return activeleafnodes
    def getActiveLeafNodesNoDummy(self):
        activeleafnodes = [x for x in self.measGraph.nodes() if self.measGraph.out_degree(x)==0 and self.measGraph.nodes[x]['status']=='Active' ] #and self.measGraph.in_degree(x)==1
        
        return activeleafnodes
    
    def getInActiveLeafNodes(self):
        Inactiveleafnodes = [x for x in self.measGraph.nodes() if self.measGraph.out_degree(x)==0 and self.measGraph.nodes[x]['status']=='InActive' ] #and self.measGraph.in_degree(x)==1
        
        return Inactiveleafnodes
    
    def isTrueBranchLnode(self,lnode):
        targctr,N,iodcnt,odcnt,meascnt,targs,targMeasSeq=self.getleafNodeMetrics(lnode,printit=False)
        
        return len(targctr)==1
    
    def getEstimateMetrics(self,Data,Tvec,planet,clusterWidths={'a':10,'e':0.1,'i':10*np.pi/180,'Om':10*np.pi/180,'om':10*np.pi/180,'f':20*np.pi/180}):
        orbs = self.getActiveOrbs(Tvec,planet,leafNodes=None)
        L=[]
        Tk = sorted(list(self.history_M.keys()))
        for targID in Data.gettargetIdxs():
            K=[]
            Nmeas=0
            for k in Tk:
                for i in range(len(self.history_M[k])):
                    if self.history_M[k][i].targID==targID:
                        K.append(k)
                        Nmeas+=1
            truorb = Data.true_catalogue.loc[targID,:]
            Nestorbs=0
            for j in range(len(orbs)):
                err = np.abs(orbs[j][:5]-truorb[:5])
                if err[0]<clusterWidths['a'] and err[1]<clusterWidths['e'] and err[2]<clusterWidths['i'] and err[3]<clusterWidths['Om'] and err[4]<clusterWidths['om']:
                    Nestorbs+=1
            
            if len(K)==0:
                L.append({'Target':targID,'meas_k0':np.nan,'meas_kf':np.nan,
                      'Nmeas':Nmeas,'Nestorbs':Nestorbs})
            else:
                L.append({'Target':targID,'meas_k0':min(K),'meas_kf':max(K),
                      'Nmeas':Nmeas,'Nestorbs':Nestorbs})
        df = pd.DataFrame(L)
        # print(df.sort_values(by='Nmeas',ascending=False))
        print(df)
        return df
    
    
    def makeNodeInActive(self,node,reason):
        if node in self.measGraph.nodes:
            if self.isTrueBranchLnode(node):
                print("############# InActivating node from true branch***********")
                
            self.measGraph.nodes[node]['status']='InActive'
            self.measGraph.nodes[node]['status_reason']=reason
        
    def getleafNodeMetrics(self,lnode,printit=True):
        brch = self.getCompleteBranch(lnode)
        N=len(brch)
        iodcnt=0
        odcnt=0
        meascnt=0
        targs=[]
        targMeasSeq=[]
        for i in range(len(brch)):
            k = self.measGraph.nodes[brch[i]]['timek']
            if self.measGraph.nodes[brch[i]].get('iod_rv',None) is not None:
                iodcnt+=1
            if self.measGraph.nodes[brch[i]].get('od_rv',None) is not None:
                odcnt+=1
            if self.measGraph.nodes[brch[i]].get('meas_idx',None) is not None:
                meascnt+=1
                measidx = self.measGraph.nodes[brch[i]].get('meas_idx')
                if self.history_M[k][measidx].targID is not None:
                    targs.append("trg:%d"%self.history_M[k][measidx].targID)
                    targMeasSeq.append((k,self.history_M[k][measidx].targID))
            else:
                targMeasSeq.append((k,None))
                
        ctr = clc.Counter(targs)
        if printit:
            print(lnode,self.measGraph.nodes[lnode]['status'],"len(brch) = ",N," targs=",clc.Counter(targs)," iodcnt=",iodcnt, " odcnt=",odcnt," meascnt=",meascnt,self.measGraph.nodes[lnode].get('od_score',None))
        return ctr,N,iodcnt,odcnt,meascnt,targs,targMeasSeq
    
    def getdebugMetrics(self):
        print("#Acive nodes = ",len(self.getActiveLeafNodes()))
        print("#In-Active nodes = ",len(self.getInActiveLeafNodes()))
        K=max(self.history_M.keys())
        TT=[(0,clc.Counter())]
        for k in range(K):
            M = self.history_M.get(k,None)
            if M is None:
                continue
            targs=[]
            for i in range(len(M)):
                targs.append("trg:%d"%self.history_M[k][i].targID)
            targs = clc.Counter(targs)
            if len(targs)>0:
                TT.append((k,+TT[-1][1]+targs))
        print(TT[-2:])
        
        leafnodes = self.getActiveLeafNodes()+self.getInActiveLeafNodes()
        # iod, od stats
        for lnode in leafnodes:    
            brch = self.getCompleteBranch(lnode)
            N=len(brch)
            iodcnt=0
            odcnt=0
            meascnt=0
            targs=[]
            for i in range(len(brch)):
                k = self.measGraph.nodes[brch[i]]['timek']
                if self.measGraph.nodes[brch[i]].get('iod_rv',None) is not None:
                    iodcnt+=1
                if self.measGraph.nodes[brch[i]].get('od_rv',None) is not None:
                    odcnt+=1
                if self.measGraph.nodes[brch[i]].get('meas_idx',None) is not None:
                    meascnt+=1
                    measidx = self.measGraph.nodes[brch[i]].get('meas_idx')
                    if self.history_M[k][measidx].targID is not None:
                        targs.append("trg:%d"%self.history_M[k][measidx].targID)
            ctr = clc.Counter(targs)
            if len(ctr)==1:
                print(lnode,self.measGraph.nodes[lnode]['status'],"len(brch) = ",N," targs=",clc.Counter(targs)," iodcnt=",iodcnt, " odcnt=",odcnt," meascnt=",meascnt,self.measGraph.nodes[lnode].get('od_score',None))
                
            # for key,val in ctr.items():
            #     if val > 0.9*meascnt and meascnt>5 and 'trg:0' in key:
            #         print(lnode,self.measGraph.nodes[lnode]['status'],"len(brch) = ",N," targs=",clc.Counter(targs)," iodcnt=",iodcnt, " odcnt=",odcnt," meascnt=",meascnt,self.measGraph.nodes[lnode].get('od_score',None))
            #         pass
                
    def add_measurements(self,k,Tvec,M,mu):
        """
        M can be a list of unit vectors (angles) or full position measurements
        
        M=[(sensID zk k targID nodeID measID),(sensID zk k targID nodeID measID),..]
        (s1,'a',L) where s1 is the sensor ID, 'a' indicates angle measurement, L is teh actual measurememt
        Parameters
        ----------
        M : list of measurements
        
        if a node is InActive, do not add anymore measurements as it is a deadend
        """
        
        
        
        self.history_M[k]=M
        
        cnt=0
            
        
        # predecessors
        # leafnodes=[nID for nID in list(self.measGraph.nodes) if self.measGraph.nodes[nID]['treestate']=='leaf']
        leafnodes = self.getActiveLeafNodes()
        
        for lnodeID in leafnodes:
            # add meas as branches
            for i in range(len(M)):
                # sensID=M[i][0]
                # sensor = [ss for ss in sensors.itersensors() if ss.idx==sensID][0]
                
                nodeID = (k,cnt)
                cnt += 1
                self.measGraph.add_node(nodeID,status='Active',timek=k,meas_idx = i,sensID=M[i].sensID,mk=None,Pk=None)
                self.measGraph.add_edge(lnodeID,nodeID)
            
                
                
            # add null node to account for missed detections
            # nodeID = uuid.uuid4()
            nodeID = (k,cnt)
            cnt += 1
            self.measGraph.add_node(nodeID,status='Active',timek=k,meas_idx = None, sensID=None,mk=None,Pk=None)
            self.measGraph.add_edge(lnodeID,nodeID)
                
        # also add the measurements as new measurements
        for i in range(len(M)):
            # nodeID = uuid.uuid4()
            nodeID = (k,cnt)
            cnt += 1
            self.measGraph.add_node(nodeID,status='Active',timek=k,meas_idx = i,sensID=M[i].sensID,mk=None,Pk=None)
            
    def getOverlappedBranches(self,mainlnode):
        if mainlnode not in self.measGraph.nodes:
            return []
            
        mainbr = self.getCompleteBranch(mainlnode)
        mainbrmeas = self.getMeasIds(mainbr)
        
        LeafNodes = self. getActiveLeafNodes()
        overlapNodes=[]
        for lnode in LeafNodes:
            br = self.getCompleteBranch(lnode)
            brmeas = self.getMeasIds(br)
            if mainbrmeas.issubset(brmeas) or brmeas.issubset(mainbrmeas):
                overlapNodes.append(lnode)    
        
        return overlapNodes
        
    def getMeasFromNodes(self,nodes):
        # return M=[(s1,'a',L,k,nodeID),(s2,'ra',r,k,nodeID),..]
        MM=[]
        for nn in nodes:
            k = self.measGraph.nodes[nn]['timek']
            i = self.measGraph.nodes[nn]['meas_idx']
            if i is not None:
                mm = self.history_M[k][i]
                mm2=ssatarg.Meas(mm.sensID, mm.zk, mm.k, mm.targID, nn, (k,i))
                # mm.append(nn) # add the node also for any back refernce
                MM.append(mm2)
            else:
                MM.append(None)
                
        return MM
            
    def remove_branch_starting_from(self,nodeID,makeInActive=False):
        """
        Removes the complete branch
        Parameters
        ----------
        nodeID

        """     
        if nodeID not in self.measGraph.nodes:
            return None
        
        if self.isTrueBranchLnode(nodeID):
            print("************Deleting node from true branch***********")
        
        N = nx.descendants(self.measGraph, nodeID)
        if makeInActive:
            for n in N:
                self.makeNodeInActive(n, ['remove branch starting from',nodeID])
            
            self.makeNodeInActive(nodeID, ['remove branch starting from',nodeID])

        else:
            self.measGraph.remove_nodes_from(N)
            self.measGraph.remove_node(nodeID)
    
    def getMeasIds(self,branch) :
        MMeas= set([(self.measGraph.nodes[bnode].get('timek',None),self.measGraph.nodes[bnode].get('meas_idx',None)) for bnode in branch if self.measGraph.nodes[bnode]['meas_idx'] is not None])
        return MMeas
        
    def isolate_branch(self,leafnode,makeInActive=False):
        """
        isolates the complete branch ending in leafnode
        This is done by removing all side branches that are connected to this brach

        """
        
        # pdb.set_trace()
        if leafnode not in self.measGraph.nodes:
            return None
        

        reqbranch = self.getCompleteBranch(leafnode)
        for i in range(len(reqbranch)-1,-1,-1):
            bnode = reqbranch[i]    
            sNodes = list(self.measGraph.successors(bnode))
            for snode in sNodes:
                if snode not in reqbranch:
                    self.remove_branch_starting_from(snode,makeInActive=makeInActive)
        
        MMreq_set = self.getMeasIds(reqbranch)
        
        
        # iterate over branches
        # leafnodes = self.getActiveLeafNodes()
        # for lnode in leafnodes:
        #     if lnode == leafnode:
        #         continue
        #     if lnode not in self.measGraph.nodes:
        #         continue
            
        #     brch = self.getCompleteBranch(lnode)
        #     measids = self.getMeasIds(brch)
        #     if measids.issubset(MMreq_set) or MMreq_set.issubset(measids):
        #         if self.measGraph.nodes[brch[-1]].get('od_rv',None) is None:
        #             continue
        #     if len(measids.intersection(MMreq_set))>0 :
        #         for i in range(len(brch)):
        #             if brch[i] not in self.measGraph.nodes:
        #                 continue
        #             mmid = self.measGraph.nodes[brch[i]]['meas_idx']
        #             tk = self.measGraph.nodes[brch[i]]['timek']
        #             if mmid is None:
        #                 continue
        #             if (tk,mmid) in MMreq_set:
        #                 parentnode = list(self.measGraph.predecessors(brch[i]))
        #                 if len(parentnode)>0:
        #                     self.measGraph.nodes[parentnode[0]]['status'] = 'InActive'
        #                     self.measGraph.nodes[parentnode[0]]['status_reason'] = ['isolated by leafnode',leafnode]
                        
        #                 self.remove_branch_starting_from(brch[i],makeInActive=makeInActive)
            
        # now make the branch measurements unique, this means , all brnahce in any tree that intersect 
        # with this branch have to be removed
        
        
        allnodes = list(self.measGraph.nodes.keys())
        for nodeID in allnodes:
            if nodeID not in self.measGraph.nodes:
                continue
                
            if nodeID in reqbranch:
                continue
            
            
                
            if self.measGraph.nodes[nodeID]['meas_idx'] is None:
                continue
            
            # if self.measGraph.nodes[nodeID].get('od_rv',None) is None:
            #     continue
            
            measID = (self.measGraph.nodes[nodeID].get('timek',None),self.measGraph.nodes[nodeID].get('meas_idx',None))
            
            if measID in MMreq_set:
                parentnode = list(self.measGraph.predecessors(nodeID))
                if len(parentnode)>0:
                    self.measGraph.nodes[parentnode[0]]['status'] = 'InActive'
                    self.measGraph.nodes[parentnode[0]]['status_reason'] = ['isolated by leafnode',leafnode]
                self.remove_branch_starting_from(nodeID,makeInActive=makeInActive)

    def iterate_last_K_step_active_branches(self,K=-1):
        """
        K=-1 means all the way up to the root nodes

        """
        leafnodes  = self.getActiveLeafNodes()
        for lnode in leafnodes:
            k=0
            L=[lnode]
            while True:
                parentnode = list(self.measGraph.predecessors(L[-1]))
                if len(parentnode)==0 or k==K:
                    # we have reached the root node or completed K steps
                    break
                else:
                    pnode = parentnode[0]
                    L.append(pnode)
                k=k+1
            
            yield L[::-1] # reverse and return (yield) it
            # reverse as we want in increasing time steps order
    
    
    
    def getCompleteBranch(self,leafnode):
        """
        K=-1 means all the way up to the root nodes
        
        """
        
        L=[leafnode]
        while True:
           parentnode = list(self.measGraph.predecessors(L[-1]))
           if len(parentnode)==0:
               # we have reached the root node or completed K steps
               break
           else:
               pnode = parentnode[0]
               L.append(pnode)
           
           
        return L[::-1] # reverse and return (yield) it
            
    def fuse_branches(self,br1,br2):
        """
        branches can be fused when there the NONE nodes of one branch can be replaced with good measurements from another branchs
        """
        
        measids1 = self.getMeasIds(br1)
        measids2 = self.getMeasIds(br2)
        if len(measids1)==len(measids2):
            s1 = self.measGraph.nodes[ br1[-1] ].get('od_score',None)
            s2 = self.measGraph.nodes[ br2[-1] ].get('od_score',None)
            print(s1,s2)
            if s1>=s2:
                brmain = br1
                br = br2
            elif s1<s2:
                brmain = br2
                br = br1

                
                
        elif len(measids1)>len(measids2):
            brmain = br1
            br = br2
        else:
            raise Exception("Case not covered for fusing")
            brmain = br2
            br = br1
      
        # brmainMs = self.getMeasIds(brmain)
        # brMs = self.getMeasIds(br)
        
        # for i in range(len(brmain)):
        #     tkmain = self.measGraph.nodes[brmain[i]].get('timek', None)
        #     if self.measGraph.nodes[brmain[i]].get('meas_idx', None) is None:
        #         for j in range(len(br)):
        #             if self.measGraph.nodes[br[j]].get('timek', None)==tkmain:
        #                 if self.measGraph.nodes[br[j]].get('meas_idx', None) is not None:
        #                     for key in self.measGraph.nodes[br[j]].keys():
        #                         self.measGraph.nodes[brmain[i]][key] = copy.deepcopy(self.measGraph.nodes[br[j]][key])
                        
        if self.isTrueBranchLnode(br[-1]):
            print("************Deleting node from true branch***********")
            
        self.measGraph.nodes[br[-1]]['status'] = 'InActive'
        self.measGraph.nodes[br[-1]]['status_reason'] = ['fuse branch brmain',brmain,br]
        return brmain

    def branchoptimization(self):
        leafNodes = self.getActiveLeafNodes()
        X=[]
        for lnode in leafNodes:
            best_score = self.measGraph.nodes[lnode].get('best_score', None)
            if best_score is None:
                continue
            branch = self.getCompleteBranch(lnode)
            MMeasIds = self.getMeasIds(branch) 
            X.append((best_score,lnode,MMeasIds))
        
        
        m = gp.Model("matrix1")

        # Create variables
        x = m.addMVar(shape=len(X), vtype=GRB.BINARY, name="x")
    
        # Set objective
        obj = np.array([-l[0] for l in X])
        m.setObjective(obj @ x, GRB.MAXIMIZE)
    
        # Build (sparse) constraint matrix
        val=[]
        row=[]
        col=[]
        c=0
        for i in range(len(X)):
            for j in range(len(X)):
                if j>i:
                    if len(X[i][2].intersection(X[j][2]))>0:
                        val.append(1)
                        col.append(j)
                        row.append(c)
                        
                        val.append(1)
                        col.append(i)
                        row.append(c)
                        
                        c+=1

    
        A = sp.csr_matrix((val, (row, col)), shape=(c, len(X)))
    
        # Build rhs vector
        lhs = np.ones(c)
    
        # Add constraints
        m.addConstr(A @ x == lhs, name="c")
    
        # Optimize model
        m.optimize()
        
        Leafopt = [X[i][1] for i in range(len(X)) if x.X[i]==1 ]
        
        for i in range(len(Leafopt)):
            self.isolate_branch(Leafopt[i],makeInActive=False)
            
            
        print(x.X)
        print('Obj: %g' % m.ObjVal)
                   
    
    def getActiveOrbs(self,Tvec,planet,leafNodes=None):
        if leafNodes is None:
            leafNodes = self.getActiveLeafNodes()
        globalOrbs = []
        Lorb=[]
        for lnode in leafNodes:
            score = self.measGraph.nodes[lnode].get('od_score', None)
            tlast = Tvec[self.measGraph.nodes[lnode].get('timek', None)]
            rv = self.measGraph.nodes[lnode].get('od_rv', None)
            if score is not None:
                # if score < 0.5:
                orb = coords.from_RV_to_OE(rv[0:3], rv[3:], planet.mu)
                # if orb[0]>14000:
                #     break
                globalOrbs.append(np.hstack([orb,tlast,score,lnode]))
                Lorb.append(lnode)
        if len(globalOrbs)>0:
            globalOrbs = np.vstack(globalOrbs)
        else:
            globalOrbs=np.array([])
        return globalOrbs
    
    