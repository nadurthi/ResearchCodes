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
#include <nlohmann/json.hpp>

class TargetModel;

class CarModel: public TargetModel{
public:
  CarModel();
  Eigen::VectorXf propforward(Eigen::VectorXf x);
  Eigen::VectorXf randinit(Eigen::VectorXf lb,Eigen::VectorXf ub);

  int dim;
  Eigen::VectorXf Q;

};
