#pragma once
#include "base.h"
#include "pcl_helpers.h"
#include "binmatch.h"
#include "localize.h"

struct BMatchAndCorrH {
        std::vector<BinMatchSol> sols;
        Eigen :: Matrix4f gHk_corr;
}

class MapLocalizer {
public:
MapLocalizer(std::string opt );
void setOptions(std::string optstr);
void resetH();

//-----------------Setters------------------
void addMeas(const Eigen::Ref<const Eigen::MatrixXf> &X,float t);
void addMap(const Eigen::Ref<const Eigen::MatrixXf> &X);
void addMap2D(const Eigen::Ref<const MatrixX2f> &X);
void setHlevels();
void setgHk(int tk, Eigen::Matrix3f gHk );
void setLookUpDist();
void setRegisteredSeqH();
void setRelStates();

//-------------------Getters------------
Eigen::MatrixXf MapLocalizer::getmeas_eigen(int k);
pcl::PointCloud<pcl::PointXYZ>::ConstPtr MapLocalizer::getmeas(int k);

float getdt();
Vector6f MapPcllimits();

Eigen::MatrixXf getmaplocal_eigen(Eigen::Vector3f lb,Eigen::Vector3f ub);

pcl::PointCloud<pcl::PointXYZ>::ConstPtr
MapLocalizer::getmaplocal(Eigen::Vector3f lb,Eigen::Vector3f ub)

Eigen::MatrixXf getmap_eigen();
pcl::PointCloud<pcl::PointXYZ>::ConstPtr getmap();


Eigen::MatrixXf MapLocalizer::getmap2D_eigen();

pcl::PointCloud<pcl::PointXYZ>::ConstPtr getmap2D();

MatrixX3f getvelocities();
MatrixX3f getpositions();
MatrixX3f getangularvelocities();

VectorXf getLikelihoods(const Eigen::Ref<const Eigen::MatrixXf> &Xposes);

std::vector<Eigen::Matrix4f> getSeq_gHk();

std::vector<Eigen::Matrix4f> MapLocalizer::getsetSeq_gHk(int t0,int tf,int tk, Eigen::Matrix4f gHk);

pcl::PointCloud<pcl::PointXYZ>::ConstPtr MapLocalizer::getalignSeqMeas(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);
MatrixX3f MapLocalizer::getalignSeqMeas_eigen(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);

//-----------------Aligners-------------------------



BMatchAndCorrH
BMatchseq(int t0,int tf,int tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHk,bool gicp=true);


// gHk takes k-frame local to gloal inertial frame
Eigen :: Matrix4f
gicp_correction(int tk, const Eigen::Ref<const Eigen :: Matrix4f>&gHk_est);



//--------------------------

int tk;
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> meas;
std::vector<float> T;
MatrixX3f XseqPos,Vel,AngVel;
std::map<std::pair<int,int>, Eigen::Matrix4f> i1Hi_seq;   // i to i+1
json options;
pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;

std::vector<Eigen::Matrix4f> gHk;

pcl::PointCloud<pcl::PointXYZ>::Ptr map;
pcl::PointCloud<pcl::PointXYZ>::Ptr map2D;
pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree;
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

MatrixXXuint16 Xdist;


BinMatch bm;
};
