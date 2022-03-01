#pragma once

#include "base.h"


void
plotpcd(pcl::visualization::PCLVisualizer::Ptr viewer,
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        const std::string& handleID,std::vector<int> color,
        int pointsize);

void
plotpoints(pcl::visualization::PCLVisualizer::Ptr viewer,
           const Eigen::Ref<const Eigen::MatrixXf> &points,
           const std::string& handleID,std::vector<int> color,
           int pointsize);


void
plotpointarrows(pcl::visualization::PCLVisualizer::Ptr viewer,
                const Eigen::Ref<const Eigen::MatrixXf> &points,
                const Eigen::Ref<const Eigen::MatrixXf> &dirs,
                const std::string& handleID,std::vector<int> color,
                int pointsize,float arrowlen);


void
plottraj(pcl::visualization::PCLVisualizer::Ptr viewer,
         const Eigen::Ref<const Eigen::MatrixXf> &points,
         const std::string& handleID,std::vector<int> color,int pointsize);




class KittiPlot {
public:
KittiPlot(std::string opt);
void update();
void clear();
void createviewer();
void plotMap(pcl::PointCloud<pcl::PointXYZ>::Ptr C1);
void plottrajectories(const Eigen::Ref<const Eigen::MatrixXf> &points);
void plotPFpoints(const Eigen::Ref<const Eigen::MatrixXf> &points,
                  const Eigen::Ref<const Eigen::MatrixXf> &dirs);

pcl::visualization::PCLVisualizer::Ptr viewer;
json options;
};
// viewer->setBackgroundColor (0, 0, 0);
// viewer->addCoordinateSystem (1.0);
