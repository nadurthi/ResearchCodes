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

#include <iostream>
#include <iomanip>
#include <map>
#include <random>


class PF{
public:
PF(std::string opt);

void propforward(){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d{0,1};

  #pragma omp parallel for num_threads(6)
  for(int i=0;i<X.rows();++i){
    Eigen::VectorXd r(X.cols());
    for(int j=0;j<X.cols();++j){
      r(j)=d(gen);
    }
    X.row(i)=model.propforward(X.row(i))+r;
  }
}

void measUpdt(Eigen::VectorXf likelihood ){
  for(int i=0;i<X.rows();++i){
    W(i)=W(i)*likelihood(i);
  }
  renormalize();
}

void renormalize(){
  W=W/W.sum();
}

void bootstrapresample(){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(W.begin(),W.end());
  for(int i=0;i<X.rows();++i){
    W(i)=d(gen);
  }
}
void Neff(){
  return 1/W.pow(2).sum();
}

json options;
Eigen::MatrixXf X;
Eigen::VectorXf W;
TargetModel model;
};
