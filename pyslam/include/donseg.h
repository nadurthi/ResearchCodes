#pragma once

#include "base.h"

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector6f = Eigen::Matrix<float, 6, 1>;
using Vector4f = Eigen::Matrix<float, 4, 1>;

using json = nlohmann::json;

namespace py = pybind11;

class Don{
public:

  Don(std::string opt);

  void setMapX(const Eigen::Ref<const Eigen::MatrixXf> &MapX);

  void computeNormals(std::string opt);

  void  computeDon(std::string opt);

  std::vector<std::pair<std::string,Eigen::MatrixXf>>
  filter(std::string opt);


  pcl::PointCloud<pcl::PointNormal>::Ptr doncloud;
  pcl::search::Search<pcl::PointXYZ>::Ptr tree;
  pcl::PointCloud<pcl::PointXYZ>::Ptr mapX;
  pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale;
  pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale;
  // pcl::PointCloud<pcl::PointXYZ>::Ptr measX;
  json options;
  // pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  // pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;



};



std::vector<std::pair<std::string,Eigen::MatrixXf>>
donsegmentation(std::string opt,const Eigen::Ref<const Eigen::MatrixXf> &Xmap);
