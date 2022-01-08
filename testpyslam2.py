import numpy as np
from pyslam import slam 
import json


D={"icp":{},
   "gicp":{},
   "gicp_cost":{},
   "ndt":{}}

D["icp"]["enable"]=0
D["icp"]["setMaximumIterations"]=500
D["icp"]["setMaxCorrespondenceDistance"]=10
D["icp"]["setRANSACIterations"]=0.0
D["icp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["icp"]["setTransformationEpsilon"]=1e-9
D["icp"]["setEuclideanFitnessEpsilon"]=0.01


D["gicp_cost"]["enable"]=1


D["gicp"]["enable"]=0
D["gicp"]["setMaxCorrespondenceDistance"]=50
D["gicp"]["setMaximumIterations"]=20.0
D["gicp"]["setMaximumOptimizerIterations"]=20.0
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


X11=np.random.randn(1000,3)


X22=X11+1


Hpcl=slam.registrations(X22,X11,json.dumps(D))
print(Hpcl)