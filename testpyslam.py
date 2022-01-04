
import numpy as np
import open3d as o3d
from pyslam import slam
# dtype = np.float64

# folder='lidarprocessing/'

# i=1
# f1="%06d.bin"%i
# X1 = np.fromfile(folder+'/'+f1, dtype=np.float32)
# X1=X1.reshape((-1, 4))
# X1=X1.astype(dtype)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(X1[:,:3])
# voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)
# X11=np.asarray(voxel_down_pcd.points)
# X11=np.ascontiguousarray(X11,dtype=dtype)

X11=np.random.randn(100,3)
X22=X11+1


D={}
D["icp_setMaximumIterations"]=500
D["icp_setMaxCorrespondenceDistance"]=10
D["icp_setRANSACIterations"]=0
D["icp_setRANSACOutlierRejectionThreshold"]=1.5
D["icp_setTransformationEpsilon"]=1e-9
D["icp_setEuclideanFitnessEpsilon"]=0.01


D["gicp_setMaxCorrespondenceDistance"]=50
D["gicp_setMaximumIterations"]=100
D["gicp_setMaximumOptimizerIterations"]=100
D["gicp_setRANSACIterations"]=0
D["gicp_setRANSACOutlierRejectionThreshold"]=1.5
D["gicp_setTransformationEpsilon"]=1e-9
D["icp_setUseReciprocalCorrespondences"]=0.1

D["ndt_setTransformationEpsilon"]=1e-9
D["ndt_setStepSize"]=2
D["ndt_setResolution"]=1
D["ndt_setMaximumIterations"]=25
D["ndt_initialguess_axisangleA"]=0
D["ndt_initialguess_axisangleX"]=0
D["ndt_initialguess_axisangleY"]=0
D["ndt_initialguess_axisangleZ"]=1
D["ndt_initialguess_transX"]=0.5
D["ndt_initialguess_transY"]=0.01
D["ndt_initialguess_transZ"]=0.01

Hpcl=slam.registrations(X22,X11,D)