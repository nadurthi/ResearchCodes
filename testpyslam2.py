import numpy as np
from pyslam import slam 
import json
import open3d as o3d
import time 
import pickle
from joblib import dump, load
from lidarprocessing.numba_codes import gicp


def pose2Rt(x):
    t=x[0:3]
    phi=x[3]
    xi=x[4]
    zi=x[5]
    
    Rzphi,dRzdphi=gicp.Rz(phi)
    Ryxi,dRydxi=gicp.Ry(xi)
    Rxzi,dRxdzi=gicp.Rx(zi)
    
    R = Rzphi.dot(Ryxi)
    R=R.dot(Rxzi)
    
    return R,t


basedir ='/media/na0043/misc/DATA/KITTI/odometry/dataset'
# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
loop_closed_seq = ['02','05','06','08']
sequence = '05'




X11=np.random.randn(1000,3)


X22=X11+1


# Hpcl=slam.registrations(X22,X11,json.dumps(D))
# print(Hpcl)



D={"icp":{},
   "gicp":{},
   "gicp_cost":{},
   "ndt":{},
   "sig0":0.5,
   "dmax":50}

D["icp"]["enable"]=0
D["icp"]["setMaximumIterations"]=5
D["icp"]["setMaxCorrespondenceDistance"]=25
D["icp"]["setRANSACIterations"]=0.0
D["icp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["icp"]["setTransformationEpsilon"]=1e-3
D["icp"]["setEuclideanFitnessEpsilon"]=1


D["gicp_cost"]["enable"]=0


D["gicp"]["enable"]=1
D["gicp"]["setMaxCorrespondenceDistance"]=20
D["gicp"]["setMaximumIterations"]=10.0
D["gicp"]["setMaximumOptimizerIterations"]=10.0
D["gicp"]["setRANSACIterations"]=0.0
D["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["gicp"]["setTransformationEpsilon"]=1e-9
D["gicp"]["setUseReciprocalCorrespondences"]=1

D["ndt"]["enable"]=0
D["ndt"]["setTransformationEpsilon"]=1e-9
D["ndt"]["setStepSize"]=2.0
D["ndt"]["setResolution"]=1.0
D["ndt"]["setMaximumIterations"]=25.0
D["ndt"]["initialguess_axisangleA"]=0.0
D["ndt"]["initialguess_axisangleX"]=0.0
D["ndt"]["initialguess_axisangleY"]=0.0
D["ndt"]["initialguess_axisangleZ"]=1.0
D["ndt"]["initialguess_transX"]=0.5
D["ndt"]["initialguess_transY"]=0.01
D["ndt"]["initialguess_transZ"]=0.01

dmax=D["dmax"]
sig0=D["sig0"]

sktree,neigh=load( "kitti-pcd-seq-%s-TREES.joblib"%sequence)

pcd=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)

pclloc=slam.Localize(json.dumps(D))

Xmap_down = np.asarray(pcd.voxel_down_sample(voxel_size=1).points)
pclloc.setMapX(Xmap_down)


with open("testpcllocalize.pkl","rb") as FF:
    Xpf,X1v_down=pickle.load(FF)

print("Xpf.shape = ",Xpf.shape)
print("X1v_down.shape = ",X1v_down.shape)

st=time.time()
ret=pclloc.computeLikelihood(Xpf,X1v_down)
et=time.time()
print("time taken for likelihood = ",et-st)
ret=dict(ret)
ret['likelihood']
# ret['Xposes_corrected']

LiK=[]
st=time.time()
for j in range(Xpf.shape[0]):
    
    # mm=robotpf.X[j][:3] #measModel(robotpf.X[j])
    # ypdf = multivariate_normal.pdf(Xtpath[k,:3], mean=mm, cov=Rposnoise)
    # robotpf.wts[j]=(1e-6+ypdf)*robotpf.wts[j]
    
    # print("j=",j)
    Dv=[]
    R,t = pose2Rt(Xpf[j][:6])
    
    X1gv_pose = R.dot(X1v_down.T).T+t
    
    
    # bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(X1gv_pose,axis=0)-5,max_bound=np.max(X1gv_pose,axis=0)+5)
    # pcdbox=pcddown.crop(bbox3d)
    
    # pcd_tree_local = o3d.geometry.KDTreeFlann(pcdbox)

    
    # Xmap=np.asarray(pcdbox.points)
    # for i in range(X1gv_pose.shape[0]):
    #     [_, idx, d] = pcd_tree.search_hybrid_vector_3d(X1gv_pose[i], dmax, 1)
    #     if len(idx)>0:
    #         Dv.append(d[0])
    #     else:
    #         Dv.append(dmax)
    # Dv=np.array(Dv)
    
    
    ddsktree,idxs = sktree.query(X1gv_pose,k=1, dualtree=False,return_distance = True)
    ddsktree=ddsktree.reshape(-1)
    idxs=idxs.reshape(-1)
    idxs=idxs[ddsktree<=dmax]
    Dv=ddsktree.copy()
    Dv[Dv>dmax]=dmax
    
    sig=D["sig0"]*np.sqrt(X1gv_pose.shape[0])
    likelihood= np.sum(Dv**2)/sig**2
    # likelihood=np.max([1e-20,likelihood])
    LiK.append(likelihood)
    # robotpf.wts[j]=likelihood*robotpf.wts[j]
    
    # Hpcl=slam.registrations(X1gv_down,X1gv_pose,json.dumps(D))
    # Hpcl=dict(Hpcl)
    # H=Hpcl["H_gicp"]
    # Hj = np.zeros((4,4))
    # Hj[0:3,0:3]=R
    # Hj[0:3,3]=t
    # Hj[3,3]=1
    
    # Hj=nplinalg.inv(Hj)
    
    # Hj=Hj.dot(H)
    # RR=Hj[0:3,0:3]
    # tt=Hj[0:3,3]
    # xpose=Rt2pose(RR,tt)
    # # pose2Rt
    # robotpf.X[j][0:6]=xpose

    
et=time.time()
print("meas model time = ",et-st)

#%%
