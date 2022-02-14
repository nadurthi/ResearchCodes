#pragma once

#include "base.h"



namespace py = pybind11;





void pose2Hmat(const Vector6d& x,Eigen::Matrix4f& H);


// void setMapX( pcl::PointCloud<pcl::PointXYZ> MapX);

VectorXf
computeLikelihood(const pcl::KdTree<pcl::PointXYZ>& mapkdtree,const Eigen::Ref<const Eigen::MatrixXf> &Xposes,pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl);

VectorXf
computeLikelihood(const pcl::octree::OctreePointCloud<pcl::PointXYZ>& mapoctree,const Eigen::Ref<const Eigen::MatrixXf> &Xposes,pcl::PointCloud<pcl::PointXYZ>::Ptr Xmeaspcl);

VectorXf
computeLikelihood_lookup(const Eigen::Ref<const MatrixXXuint16> &Xdist,const Eigen::Ref<const Eigen::MatrixXf> &Xposes,pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl);

float getNNsqrddist2Map(const pcl::octree::OctreePointCloud<pcl::PointXYZ>& mapoctree,const pcl::PointXYZ & searchPoint,float dmax);
float getNNsqrddist2Map(const pcl::KdTree<pcl::PointXYZ>& mapkdtree,const pcl::PointXYZ & searchPoint,float dmax);
