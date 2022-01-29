#pragma once

#include "base.h"



namespace py = pybind11;





void pose2Hmat(const Vector6d& x,Eigen::Matrix4f& H);

class Localize {
public:

Localize(std::string opt);
void setOptions(std::string opt);
void setMapX(const Eigen::Ref<const Eigen::MatrixXf> &MapX);
// void setMapX( pcl::PointCloud<pcl::PointXYZ> MapX);

std::vector<std::pair<std::string,Eigen::MatrixXf> >
computeLikelihood(const Eigen::Ref<const Eigen::MatrixXf> &Xposes,const Eigen::Ref<const Eigen::MatrixXf> &Xmeas);

float getNNsqrddist2Map(pcl::PointXYZ searchPoint,float dmax);

pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree;
pcl::PointCloud<pcl::PointXYZ>::Ptr mapX;
json options;
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;



};
