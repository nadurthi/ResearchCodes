#include "pcl_helpers.h"


void eigen2pcl(const Eigen::Ref<const Eigen::MatrixXf> &X,pcl::PointCloud<pcl::PointXYZ>::Ptr  & C,bool append){
								pcl::PointCloud<pcl::PointXYZ>::Ptr C1(new pcl::PointCloud<pcl::PointXYZ>(X.rows(),1));
								int i=0;
								for(auto& p: *C1) {
																p.x = X(i,0);
																p.y = X(i,1);
																p.z = X(i,2);
																++i;
								}

								if(append==false)
																C=C1;
								else{
																*C+=*C1;
								}

}

void pcl2eigen(pcl::PointCloud<pcl::PointXYZ>::Ptr  & C,Eigen :: MatrixXf& X){
								Eigen :: MatrixXf X1=Eigen :: MatrixXf::Zero(C->size(),3);
								int i=0;
								for(auto& p: *C) {
																X1(i,0) = p.x;
																X1(i,1) = p.y;
																X1(i,2) = p.z;
																++i;
								}
								X=X1;
}

void pcl_filter_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr  & inputcld,pcl::PointCloud<pcl::PointXYZ>::Ptr  & outputcld,std::vector<float> res){
								pcl::VoxelGrid<pcl::PointXYZ> sor;
								sor.setInputCloud (inputcld);
								sor.setLeafSize(res[0], res[1], res[2]);
								sor.filter (*outputcld);
}