#pragma once
#include "base.h"



void pose2Hmat_(const Vector6f& x,Eigen::Matrix4f& H);


class MeasManager {
public:
MeasManager(std::string opt );

void addMeas(const Eigen::Ref<const Eigen::MatrixXf> &X,float t);
float getdt();
void resetH();
void computeSeqH();
void registerSeqMeas();

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> meas;
std::vector<float> T;
std::map<std::pair<int,int>, Eigen::Matrix4f> i1Hi_seq;   // i to i+1
std::vector<Eigen::Matrix4f> gHs;   // ground to scan local frame to ground inertial frame
json options;
pcl::MyGeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
};

class MapManager {
public:
MapManager(std::string opt );

void addMap(const Eigen::Ref<const Eigen::MatrixXf> &X);

void addMap2D(const Eigen::Ref<const Eigen::MatrixXf> &X);
pcl::PointCloud<pcl::PointXYZ>::ConstPtr getmap();
pcl::PointCloud<pcl::PointXYZ>::ConstPtr getmap2D();

pcl::PointCloud<pcl::PointXYZ>::Ptr getmaplocal(Eigen::Vector3f lb,Eigen::Vector3f ub);
float getNNsqrddist2Map(pcl::PointXYZ searchPoint,float dmax);


pcl::PointCloud<pcl::PointXYZ>::Ptr map;
pcl::PointCloud<pcl::PointXYZ>::Ptr map2D;
pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree;
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

json options;
};

/**
   Xposes = [x,y,z, phi,xi,zi]
   phi is about z
   xi is about y
   zi is about x

 **/

Eigen::VectorXf
computeLikelihood(const MapManager &map,
                  const Eigen::Ref<const Eigen::MatrixXf> &Xposes, const Eigen::Ref<const Eigen::MatrixXf> &Xmeas);
