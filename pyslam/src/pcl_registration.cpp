#include "pcl_registration.h"



std::map<std::string, int> convert_dict_to_map(py::dict dictionary)
{
    std::map<std::string, int> result;
    for (std::pair<py::handle, py::handle> item : dictionary)
    {
        auto key = item.first.cast<std::string>();
        auto value = item.second.cast<float>();
        //cout << key << " : " << value;
        result[key] = value;
    }
    return result;
}

void print4x4Matrix (const Eigen::Matrix4f & matrix)
{
  printf ("Rotation matrix :\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}

int add(int i, int j,py::dict dict,std::string c) {
  for (auto item : dict)
    {
        std::cout << "key: " << item.first << ", value=" << item.second << std::endl;
    }

    json j_complete = json::parse(c);

    return i + j;
}

std::vector<Eigen::MatrixXf>  registrations(const Eigen::Ref<const Eigen::MatrixXf> &X1,const Eigen::Ref<const Eigen::MatrixXf> &X2,py::dict dict){
//Eigen::MatrixXf
std::vector<Eigen::MatrixXf> H;
auto D=convert_dict_to_map(dict);

pcl::PointCloud<pcl::PointXYZ>::Ptr C1(new pcl::PointCloud<pcl::PointXYZ>(X1.rows(),1));
pcl::PointCloud<pcl::PointXYZ>::Ptr C2(new pcl::PointCloud<pcl::PointXYZ>(X2.rows(),1));

pcl::PointCloud<pcl::PointXYZ>::Ptr resultXicp(new pcl::PointCloud<pcl::PointXYZ>);

int i=0;
for(auto& p: *C1){
  p.x = X1(i,0);
  p.y = X1(i,1);
  p.z = X1(i,2);
  ++i;
}
i=0;
for(auto& p: *C2){
  p.x = X2(i,0);
  p.y = X2(i,1);
  p.z = X2(i,2);
  ++i;
}

std::cout << C1->points[20].x << " " << C1->points[20].y << " " << C1->points[20].y << std::endl;
std::cout << C2->points[20].x << " " << C2->points[20].y << " " << C2->points[20].y << std::endl;

std::chrono::steady_clock::time_point begin_icp = std::chrono::steady_clock::now();

pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
icp.setMaximumIterations (D["icp_setMaximumIterations"]);
icp.setMaxCorrespondenceDistance(D["icp_setMaxCorrespondenceDistance"]); //50
icp.setRANSACIterations(D["icp_setRANSACIterations"]);
icp.setRANSACOutlierRejectionThreshold (D["icp_setRANSACOutlierRejectionThreshold"]); //1.5
icp.setTransformationEpsilon(D["icp_setTransformationEpsilon"]); //1e-9
icp.setEuclideanFitnessEpsilon(D["icp_setEuclideanFitnessEpsilon"]); //1

icp.setInputSource(C2);
icp.setInputTarget(C1);

icp.align (*resultXicp);
auto H_icp = icp.getFinalTransformation ();

std::chrono::steady_clock::time_point end_icp = std::chrono::steady_clock::now();
std::cout << "Time ICP difference = "<< std::chrono::duration_cast<std::chrono::seconds>(end_icp - begin_icp).count() << "[s]"
                                 << std::chrono::duration_cast<std::chrono::milliseconds>(end_icp - begin_icp).count() << "[ms]"
                                 << std::chrono::duration_cast<std::chrono::microseconds>(end_icp - begin_icp).count() << "[µs]" << std::endl;

// icp.hasConverged ();
// std::cout << "\nICP has converged, score is " << icp.getFitnessScore () << std::endl;
// std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;

// print4x4Matrix (H_icp);

std::chrono::steady_clock::time_point begin_gicp = std::chrono::steady_clock::now();

pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
gicp.setMaxCorrespondenceDistance(D["gicp_setMaxCorrespondenceDistance"]);
gicp.setMaximumIterations(D["gicp_setMaximumIterations"]);
gicp.setMaximumOptimizerIterations(D["gicp_setMaximumOptimizerIterations"]);
gicp.setRANSACIterations(D["gicp_setRANSACIterations"]);
gicp.setRANSACOutlierRejectionThreshold(D["gicp_setRANSACOutlierRejectionThreshold"]);
gicp.setTransformationEpsilon(D["gicp_setTransformationEpsilon"]);
gicp.setUseReciprocalCorrespondences(D["icp_setUseReciprocalCorrespondences"]); //0.1

pcl::PointCloud<pcl::PointXYZ>::Ptr resultXgicp(new pcl::PointCloud<pcl::PointXYZ>);

gicp.setInputSource(C1);
gicp.setInputTarget(C2);

gicp.align(*resultXgicp);
auto H_gicp = gicp.getFinalTransformation();

std::chrono::steady_clock::time_point end_gicp = std::chrono::steady_clock::now();
std::cout << "Time GICP difference = "<< std::chrono::duration_cast<std::chrono::seconds>(end_gicp - begin_gicp).count() << "[s]"
                                 << std::chrono::duration_cast<std::chrono::milliseconds>(end_gicp - begin_gicp).count() << "[ms]"
                                 << std::chrono::duration_cast<std::chrono::microseconds>(end_gicp - begin_gicp).count() << "[µs]" << std::endl;
// print4x4Matrix (H_gicp);

std::chrono::steady_clock::time_point begin_ndt = std::chrono::steady_clock::now();

pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
ndt.setTransformationEpsilon(D["ndt_setTransformationEpsilon"]);
ndt.setStepSize(D["ndt_setStepSize"]); //0.1
ndt.setResolution(D["ndt_setResolution"]); //1
ndt.setMaximumIterations(D["ndt_setMaximumIterations"]);

pcl::PointCloud<pcl::PointXYZ>::Ptr resultXndt(new pcl::PointCloud<pcl::PointXYZ>);

// Set initial alignment estimate found using robot odometry.
Eigen::Vector3f ax;
ax(0)=D["ndt_initialguess_axisangleX"];
ax(1)=D["ndt_initialguess_axisangleY"];
ax(2)=D["ndt_initialguess_axisangleZ"];

Eigen::Translation3f init_translation(D["ndt_initialguess_transX"],D["ndt_initialguess_transY"],D["ndt_initialguess_transZ"]);
// init_translation(0) = D["ndt_initialguess_transX"];
// init_translation(1) = D["ndt_initialguess_transY"];
// init_translation(2) = D["ndt_initialguess_transZ"];

Eigen::AngleAxisf init_rotation (D["ndt_initialguess_axisangleA"], ax);
Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

ndt.setInputSource(C1);
ndt.setInputTarget(C2);

ndt.align(*resultXndt,init_guess);
auto H_ndt = ndt.getFinalTransformation();
// print4x4Matrix (H_ndt);
std::chrono::steady_clock::time_point end_ndt = std::chrono::steady_clock::now();
std::cout << "Time NDT difference = "<< std::chrono::duration_cast<std::chrono::seconds>(end_ndt - begin_ndt).count() << "[s]"
                                 << std::chrono::duration_cast<std::chrono::milliseconds>(end_ndt - begin_ndt).count() << "[ms]"
                                 << std::chrono::duration_cast<std::chrono::microseconds>(end_ndt - begin_ndt).count() << "[µs]" << std::endl;

H.push_back(H_icp);
H.push_back(H_gicp);
H.push_back(H_ndt);

return H;

}
