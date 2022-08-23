# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as nplinalg
# from ssatoolkit import coords
# from ssatoolkit import filters
from sklearn.cluster import KMeans

def getOrbClusters(globalOrbs,clusterWidths={'a':10,'e':0.1,'i':10*np.pi/180,'Om':10*np.pi/180,'om':10*np.pi/180,'f':20*np.pi/180}):
    globalOrbs_ = np.copy(globalOrbs)
    
    ncls =int(len(globalOrbs_)/2)
    cluster_orbs = []
    while 1:
        
        
        to_do_clusters = []
        if ncls > len(globalOrbs_):
            ncls = len(globalOrbs_)
            
        kmeans = KMeans(n_clusters=ncls, random_state=0).fit(globalOrbs_[:,:6])
        newClusAdded = False
        for i in range(ncls):
            failflg=0
            pts = globalOrbs_[kmeans.labels_==i,:]
            if len(pts)==0:
                continue
            
            if np.max(pts[:,0])-np.min(pts[:,0])>clusterWidths['a']:
                failflg=1
                
            if np.max(pts[:,1])-np.min(pts[:,1])>clusterWidths['e']:
                failflg=1
                
            if np.max(pts[:,2])-np.min(pts[:,2])>clusterWidths['i']:
                failflg=1
                    
            if np.max(pts[:,3])-np.min(pts[:,3])>clusterWidths['Om']:
                failflg=1
                    
            if np.max(pts[:,4])-np.min(pts[:,4])>clusterWidths['om']:
                failflg=1
                    
            
            if np.max(pts[:,5])-np.min(pts[:,5])>clusterWidths['f']:
                failflg=1
                
            if failflg==0:
                cluster_orbs.append(pts)
                newClusAdded = True
            else:
                to_do_clusters.append(pts)
        
                    
        
        if len(to_do_clusters) == 0:
            break
        
        if newClusAdded is False:
            ncls+=20
        else:
            ncls =int(len(globalOrbs_)/2)
            
        globalOrbs_= np.vstack(to_do_clusters)
        print(ncls,len(cluster_orbs),len(globalOrbs_))        
    cluslens = np.array([len(xx) for xx in cluster_orbs])
    sortidx=np.argsort(cluslens)
    cluster_orbs = [cluster_orbs[sortidx[i]] for i in range(len(cluster_orbs))]
    return cluster_orbs[::-1]
            
def getConeOfNormals(MM,half_angle_thresh):
    """
    estimate angles betwwen normals, check if it falls within the 
    half_angle_thresh angle
    """
    zk=[]
    # get all the measurements into a vector
    for i in range(len(MM)):
        z = MM[i].zk
        zk.append(z/nplinalg.norm(z))
    
    
    # compute normals
    norms = []
    for i in range(len(MM)):
        for j in range(len(MM)):
            if i<j:
                norms.append(np.cross(zk[i],zk[j]))
    
    # now compute angle between the normals
    angles = []    
    for i in range(len(norms)):
        for j in range(len(norms)):
            if i<j:
                angles.append(np.arccos(np.dot(norms[i],norms[j])))
    
    if np.max(angles) <=  2*half_angle_thresh:
        return True
    else:
        # the estimated cone is too big 
        return False
        
        
        