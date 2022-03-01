#pragma once

#include "base.h"



namespace py = pybind11;





Eigen::Matrix4f pose2Hmat(const Vector6f& x);
Vector6f Hmat2pose(const Eigen::Matrix4f& H);

Vector6f Hmat2pose_v2(const Eigen::Matrix4f& H);

// void setMapX( pcl::PointCloud<pcl::PointXYZ> MapX);


Eigen::VectorXf
computeLikelihood(pcl::KdTree<pcl::PointXYZ>::Ptr mapkdtree,
                  const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                  pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl,float dmax,float sig0);

Eigen::VectorXf
computeLikelihood(pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr mapoctree,
                  const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                  pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl,float dmax,float sig0);
Eigen::VectorXf
computeLikelihood_lookup(const xdisttype &Xdist, const std::vector<float>& res,const std::vector<float>& Xdist_min,const std::vector<float>& Xdist_max,
                         const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                         pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl,float dmax,float sig0);

float getNNsqrddist2Map(pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr mapoctree,const pcl::PointXYZ & searchPoint,float dmax);


float getNNsqrddist2Map(pcl::KdTree<pcl::PointXYZ>::Ptr mapkdtree,const pcl::PointXYZ & searchPoint,float dmax);
