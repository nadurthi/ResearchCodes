#pragma once

#include "base.h"
#include "pcl_helpers.h"
#include "binmatch.h"
#include "localize.h"

struct BMatchAndCorrH {
        std::vector<BinMatchSol> sols;
        Eigen :: Matrix4f gHkcorr;
};

class MapLocalizer {
public:
MapLocalizer(std::string opt );
void setOptions(std::string optstr);
void resetH();

//-----------------Setters------------------
void addMeas(const Eigen::Ref<const Eigen::MatrixXf> &X,const Eigen::Ref<const Eigen::MatrixXf> &Xnoroad,float t);


void addMap(const Eigen::Ref<const Eigen::MatrixXf> &X);
void addMap2D(const Eigen::Ref<const Eigen::MatrixXf> &X);

void setgHk(int tk, Eigen::Matrix4f gHk );
void setLookUpDist();
void setRegisteredSeqH();
std::vector<Eigen::Matrix4f> setSeq_gHk();
void setRelStates();

//-------------------Getters------------
Eigen::MatrixXf getmeas_eigen(int k);
pcl::PointCloud<pcl::PointXYZ>::ConstPtr getmeas(int k);
pcl::PointCloud<pcl::PointXYZ>::ConstPtr getmeas_noroad(int k);

float getdt();
Vector6f MapPcllimits();

pcl::PointCloud<pcl::PointXYZ>::Ptr
getmaplocal(Eigen::Vector3f lb,Eigen::Vector3f ub);

pcl::PointCloud<pcl::PointXYZ>::Ptr
getmaplocal(pcl::PointXYZ min_pt,pcl::PointXYZ max_pt);

Eigen::MatrixXf getmaplocal_eigen(Eigen::Vector3f lb,Eigen::Vector3f ub);

Eigen::MatrixXf getmap_eigen();
pcl::PointCloud<pcl::PointXYZ>::ConstPtr getmap();


Eigen::MatrixXf getmap2D_eigen();

pcl::PointCloud<pcl::PointXYZ>::ConstPtr getmap2D();

MatrixX3f getvelocities();
MatrixX3f getpositions();
MatrixX3f getangularvelocities();

Eigen::VectorXf getLikelihoods(const Eigen::Ref<const Eigen::MatrixXf> &Xposes,int tk);

std::vector<Eigen::Matrix4f> getSeq_gHk();

std::vector<Eigen::Matrix4f> getsetSeq_gHk(int t0,int tf,int tk, Eigen::Matrix4f gHk);

pcl::PointCloud<pcl::PointXYZ>::Ptr getalignSeqMeas(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);
pcl::PointCloud<pcl::PointXYZ>::Ptr getalignSeqMeas_noroad(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);

MatrixX3f getalignSeqMeas_eigen(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);
MatrixX3f getalignSeqMeas_noroad_eigen(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);
//-----------------Aligners-------------------------



BMatchAndCorrH
BMatchseq(int t0,int tf,int tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHk,bool gicp=true);


// gHk takes k-frame local to gloal inertial frame
Eigen :: Matrix4f
gicp_correction(pcl::PointCloud<pcl::PointXYZ>::Ptr Xsrcpcl, const Eigen::Ref<const Eigen :: Matrix4f>&gHk_est);



//--------------------------

int tk;
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> meas,meas_noroad;
std::vector<float> T;
MatrixX3f XseqPos,Vel,AngVel;
std::unordered_map<int, std::unordered_map<int,Eigen::Matrix4f> > i1Hi_seq;
json options;
pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp,gicpseq;

std::vector<Eigen::Matrix4f> gHk;

pcl::PointCloud<pcl::PointXYZ>::Ptr map;
pcl::PointCloud<pcl::PointXYZ>::Ptr map2D;
pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr octree;
pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree;

std::vector<MatrixXXuint16> Xdist;


BinMatch bm;
};
