#pragma once

#include "base.h"


void
plotpcd(pcl::visualization::PCLVisualizer::Ptr viewer,
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        std::string handleID,std::vector<int> color,
        int pointsize){


  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, color[0], color[1], color[2]);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, handleID);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize, handleID);
}
void
plotpoints(pcl::visualization::PCLVisualizer::Ptr viewer,
        const Eigen::Ref<const Eigen::MatrixXf> &points,
        std::string handleID,std::vector<int> color,
        int pointsize){
          pcl::PointCloud<pcl::PointXYZ>::Ptr C1(new pcl::PointCloud<pcl::PointXYZ>(points.rows(),1));
          int i=0;
          for(auto& p: *C1){
            p.x = points(i,0);
            p.y = points(i,1);
            p.z = points(i,2);
            ++i;
          }

          plotpcd(viewer,C1,handleID,color,pointsize);


}
void
plotpointarrows(pcl::visualization::PCLVisualizer::Ptr viewer,
        const Eigen::Ref<const Eigen::MatrixXf> &points,
        const Eigen::Ref<const Eigen::MatrixXf> &dirs,
        std::string handleID,std::vector<int> color,
        int pointsize,float arrowlen){
        pcl::PointCloud<pcl::PointXYZ>::Ptr C1(new pcl::PointCloud<pcl::PointXYZ>(points.rows(),1));
        pcl::PointCloud<pcl::PointXYZ>::Ptr C2(new pcl::PointCloud<pcl::PointXYZ>(points.rows(),1));
        int i=0;
        for(auto& p: *C1){
          p.x = points(i,0);
          p.y = points(i,1);
          p.z = points(i,2);
          ++i;
        }
        for(std::size_t j=0;j<i;++j){
          C2->points[j].x = C1->points[j].x+arrowlen*dirs(j,0);
          C2->points[j].y = C1->points[j].y+arrowlen*dirs(j,1);
          C2->points[j].z = C1->points[j].z+arrowlen*dirs(j,2);
        }

        plotpcd(viewer,C1,handleID,color,pointsize);
        for（std::size_t nIndex = 0； nIndex < C1->points.size()-1; nIndex++）{
          viewer->addArrow (C1->points[nIndex], C2->points[nIndex], color[0], color[1], color[2], handleID);
        }
}

void
plottraj(pcl::visualization::PCLVisualizer::Ptr viewer,
  onst Eigen::Ref<const Eigen::MatrixXf> &points,
  std::string handleID,std::vector<int> color,int pointsize){
    for（std::size_t nIndex = 0； nIndex < points.rows()-1; nIndex++）{
      pcl::PointXYZ p1,p2;
      p1.x = points(nIndex,0);
      p1.x = points(nIndex,1);
      p1.x = points(nIndex,2);

      p2.x = points(nIndex+1,0);
      p2.x = points(nIndex+1,1);
      p2.x = points(nIndex+1,2);

      viewer->addLine(p1, p2,color[0], color[1],color[2], handleID);
    }
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize, handleID);
  }




class KittiPlot{
public:
  KittiPlot(std::string opt);
  void update(){
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  };
  void clear(){
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
  };
  void plotMap(pcl::PointCloud<pcl::PointXYZ>::Ptr C1){
    int r = options["plotting"]["map_color_r"];
    int g = options["plotting"]["map_color_g"];
    int b = options["plotting"]["map_color_b"];
    int pointsize = options["plotting"]["map_pointsize"];
    plotpcd(viewer, C1,"mappcd",{r,g,b},pointsize);
  };
  void plottraj(const Eigen::Ref<const Eigen::MatrixXf> &points){
    int r = options["plotting"]["traj_color_r"];
    int g = options["plotting"]["traj_color_g"];
    int b = options["plotting"]["traj_color_b"];
    int pointsize = options["plotting"]["traj_pointsize"];
    plottraj(viewer,points,"traj",{r,g,b},pointsize);
  }
  void plotPFpoints(const Eigen::Ref<const Eigen::MatrixXf> &points,
    const Eigen::Ref<const Eigen::MatrixXf> &dirs){
    int r = options["plotting"]["pf_color_r"];
    int g = options["plotting"]["pf_color_g"];
    int b = options["plotting"]["pf_color_b"];
    int pointsize = options["plotting"]["pf_pointsize"];
    float arrowlen= options["plotting"]["pf_arrowlen"];
    plotpoints(viewer,points,"pfpoints",color,pointsize);
    plotpointarrows(viewer,points,dirs,"pfdirs",color,pointsize,arrowlen);
  }

  pcl::visualization::PCLVisualizer::Ptr viewer;
  json options;
};
// viewer->setBackgroundColor (0, 0, 0);
// viewer->addCoordinateSystem (1.0);
