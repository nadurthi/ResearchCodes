#pragma once

#include "base.h"
#include "pcl_helpers.h"
#include "binmatch.h"
#include "localize.h"
#include "pcl_visual.h"

struct structmeas {
// dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem
        float dt;
        float tk;
        Eigen :: MatrixXf X1v;
        Eigen :: MatrixXf X1gv;
        Eigen :: MatrixXf X1v_roadrem;
        Eigen :: MatrixXf X1gv_roadrem;

};

struct BMatchAndCorrH {
        std::vector<BinMatchSol> sols;
        Eigen :: Matrix4f gHkcorr;
        bool isDone;
};

struct BMatchAndCorrH_async {
        BMatchAndCorrH bmHsol;
        int tk;
        int t0;
        int tf;
        bool do_gicp;
        Eigen :: Matrix4f gHkest_initial;
        Eigen :: Matrix4f gHkest_final;


        // std::chrono::time_point<std::chrono::system_clock> st;
        // std::chrono::time_point<std::chrono::system_clock> et;
        float time_taken;
        bool isDone;

};


class MapLocalizer {
public:
MapLocalizer(std::string opt );
void setOptions(std::string optstr);
void setBMOptions(std::string opt);
void resetH();
void cleanUp(int k);
void setquitsim();

void plotsim(const Eigen::Ref<const Eigen::MatrixXf> &Xpose);
void removeRoad(pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl,pcl::PointCloud<pcl::PointXYZ>::Ptr& Xpcl_noroad);
void autoReadMeas(std::string folder);
void autoReadMeas_async(std::string folder);

std::vector<Eigen::MatrixXf> getMeasQ_eigen(bool popit);

//-----------------Setters------------------
structmeas addMeas_fromQ(Eigen::Matrix4f H,float t);

void addMeas(const Eigen::Ref<const Eigen::MatrixXf> &X,const Eigen::Ref<const Eigen::MatrixXf> &Xnoroad,float t);



void addMap(const Eigen::Ref<const Eigen::MatrixXf> &X);
void addMap2D(const Eigen::Ref<const Eigen::MatrixXf> &X);

void setgHk(int tk, Eigen::Matrix4f gHk );
void setLookUpDist(std::string filename);
void setRegisteredSeqH();
void setRegisteredSeqH_async();
std::vector<Eigen::Matrix4f> setSeq_gHk();
void setRelStates();
void setRelStates_async();
void computeHlevels();

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

Eigen::MatrixXf getmap2D_noroad_res_eigen(std :: vector<float> res,int dim);
Eigen::MatrixXf getmap2D_eigen();

pcl::PointCloud<pcl::PointXYZ>::ConstPtr getmap2D();

std::vector<Eigen::Vector3f> getvelocities();
std::vector<Eigen::Vector3f> getpositions();
std::vector<Eigen::Vector3f> getangularvelocities();

Eigen::VectorXf getLikelihoods_octree(const Eigen::Ref<const Eigen::MatrixXf> &Xposes,int tk);
Eigen::VectorXf getLikelihoods_lookup(const Eigen::Ref<const Eigen::MatrixXf> &Xposes,int tk);

std::vector<Eigen::Matrix4f> getSeq_gHk();
std::vector<Eigen::Matrix4f> geti1Hi_seq();

std::vector<Eigen::Matrix4f> getsetSeq_gHk(int tk, Eigen::Matrix4f gHk);

pcl::PointCloud<pcl::PointXYZ>::Ptr getalignSeqMeas(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);
pcl::PointCloud<pcl::PointXYZ>::Ptr getalignSeqMeas_noroad(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);

Eigen :: MatrixXf getalignSeqMeas_eigen(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);
Eigen :: MatrixXf getalignSeqMeas_noroad_eigen(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim);
//-----------------Aligners-------------------------
void
BMatchseq_async(int t0,int tf,int tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHkest,bool gicp);

BMatchAndCorrH_async
getBMatchseq_async();

BMatchAndCorrH_async
BMatchseq_async_caller(int t0,int tf,int tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHkest,bool gicp);

BMatchAndCorrH
BMatchseq(int t0,int tf,int tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHk,bool gicp=true);


// gHk takes k-frame local to gloal inertial frame
Eigen :: Matrix4f
gicp_correction(pcl::PointCloud<pcl::PointXYZ>::Ptr Xsrcpcl, const Eigen::Ref<const Eigen :: Matrix4f>&gHk_est);


timerdict
gettimers();

//--------------------------

int tk;
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> meas,meas_noroad;
std::vector<float> T;
std::vector<Eigen::Vector3f> XseqPos,Vel,AngVel;
std::unordered_map<int, std::unordered_map<int,Eigen::Matrix4f> > i1Hi_seq;
json options;
pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp,gicpseq;

std::vector<Eigen::Matrix4f> gHk;

pcl::PointCloud<pcl::PointXYZ>::Ptr map;
pcl::PointCloud<pcl::PointXYZ>::Ptr map2D;
pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr octree;
pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree;

xdisttype Xdist;
std::vector<float> Xdist_min,Xdist_max;

BinMatch bm;
std::future<BMatchAndCorrH_async> bmHsols_async_future;

KittiPlot plotter;
std::queue<pcl::PointCloud<pcl::PointXYZ>::Ptr> measQ,measnoroadQ;

timerdictptr timerptr;

bool quitsim;
};
