#pragma once

#include <pybind11/eigen.h>
#include <iostream>
#include <thread>
#include <array>
#include <numeric>
#include <Eigen/Geometry>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <utility>
#include <map>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <string>
#include <algorithm>
// #include <pcl/registration/gicp.h>
#include <pcl/registration/mygicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <cmath>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/features/don.h>


#include <nlohmann/json.hpp>

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector6f = Eigen::Matrix<float, 6, 1>;
using Vector4f = Eigen::Matrix<float, 4, 1>;

using json = nlohmann::json;

namespace py = pybind11;


std::vector<std::pair<std::string,Eigen::MatrixXf>>
donsegmentation(std::string opt,const Eigen::Ref<const Eigen::MatrixXf> &Xmap);
