#pragma once
#include "base.h"

void eigen2pcl(const Eigen::Ref<const Eigen::MatrixXf> &X,pcl::PointCloud<pcl::PointXYZ>::Ptr C,bool append=false);
void pcl2eigen(pcl::PointCloud<pcl::PointXYZ>::Ptr C,Eigen::Ref< Eigen :: MatrixXf>X);

void pcl_filter_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcld,pcl::PointCloud<pcl::PointXYZ>::Ptr outputcld,std::vector<float> res);
