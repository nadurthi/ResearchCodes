#pragma once

#include "base.h"



namespace py = pybind11;





void pose2Hmat(const Vector6f& x,Eigen::Matrix4f& H);
void Hmat2pose(const Eigen::Matrix4f& H,Vector6f& x);

// void setMapX( pcl::PointCloud<pcl::PointXYZ> MapX);

VectorXf
computeLikelihood(pcl::KdTree<pcl::PointXYZ>::Ptr mapkdtree,
                  const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                  pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl,float dmax,float sig0);

VectorXf
computeLikelihood(pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr mapoctree,
                  const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                  pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl,float dmax,float sig0);
VectorXf
computeLikelihood_lookup(const xdisttype &Xdist, const std::vector<float>& res,const std::vector<float>& Xdist_min,
                         const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                         pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl,float dmax,float sig0);

float getNNsqrddist2Map(pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr mapoctree,const pcl::PointXYZ & searchPoint,float dmax);


float getNNsqrddist2Map(pcl::KdTree<pcl::PointXYZ>::Ptr mapkdtree,const pcl::PointXYZ & searchPoint,float dmax);
