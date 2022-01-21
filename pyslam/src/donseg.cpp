#include "donseg.h"
#include <omp.h>

Don::Don(std::string opt){
  options=json::parse(opt);
}

void
Don::setMapX(const Eigen::Ref<const Eigen::MatrixXf> &Xmap){
  mapX.reset(new pcl::PointCloud<pcl::PointXYZ>(Xmap.rows(),1));
  // mapX->points.resize(MapX.rows());

  int i=0;
  for(auto& p: *mapX){
    p.x = Xmap(i,0);
    p.y = Xmap(i,1);
    p.z = Xmap(i,2);
    ++i;
  }

  tree.reset (new pcl::search::KdTree<pcl::PointXYZ> (false));
  tree->setInputCloud (mapX);

  doncloud.reset (new pcl::PointCloud<pcl::PointNormal>);
}

void Don::computeNormals(std::string opt){
  options=json::parse(opt);
  ///The smallest scale to use in the DoN filter.
  double scale1=options["DON"]["scale1"];

  ///The largest scale to use in the DoN filter.
  double scale2=options["DON"]["scale2"];

  if (scale1 >= scale2)
  {
    std::cerr << "Error: Large scale must be > small scale!" << std::endl;
    exit (EXIT_FAILURE);
  }

  normals_small_scale.reset (new pcl::PointCloud<pcl::PointNormal>);
  normals_large_scale.reset (new pcl::PointCloud<pcl::PointNormal>);

  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> ne;
  ne.setInputCloud (mapX);
  ne.setSearchMethod (tree);
  /**
   * NOTE: setting viewpoint is very important, so that we can ensure
   * normals are all pointed in the same direction!
   */
  ne.setViewPoint (std::numeric_limits<float>::max (), std::numeric_limits<float>::max (), std::numeric_limits<float>::max ());
  // ne.setViewPoint (0, 0, 0);
  // calculate normals with the small scale

  ne.setRadiusSearch (scale1);
  ne.compute (*normals_small_scale);


  // calculate normals with the large scale

  ne.setRadiusSearch (scale2);
  ne.compute (*normals_large_scale);

}


void Don::computeDon(std::string opt){
  options=json::parse(opt);


  // Create output cloud for DoN results

  pcl::copyPointCloud (*mapX, *doncloud);

  // Create DoN operator
  pcl::DifferenceOfNormalsEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::PointNormal> don;
  don.setInputCloud (mapX);
  don.setNormalScaleLarge (normals_large_scale);
  don.setNormalScaleSmall (normals_small_scale);

  if (!don.initCompute ())
  {
    std::cerr << "Error: Could not initialize DoN feature operator" << std::endl;
    exit (EXIT_FAILURE);
  }

  // Compute DoN
  don.computeFeature (*doncloud);

}


std::vector<std::pair<std::string,Eigen::MatrixXf>>
Don::filter(std::string opt){
  std::vector<std::pair<std::string,Eigen::MatrixXf>> ret;
  options=json::parse(opt);

  ///The minimum DoN magnitude to threshold by
  double threshold_curv_lb=options["DON"]["threshold_curv_lb"];
  double threshold_curv_ub=options["DON"]["threshold_curv_ub"];
  double threshold_small_nz_lb=options["DON"]["threshold_small_nz_lb"];
  double threshold_small_nz_ub=options["DON"]["threshold_small_nz_ub"];
  double threshold_large_nz_lb=options["DON"]["threshold_large_nz_lb"];
  double threshold_large_nz_ub=options["DON"]["threshold_large_nz_ub"];

  ///segment scene into clusters with given distance tolerance using euclidean clustering
  double segradius=options["DON"]["segradius"];


  // Filter by magnitude
  // Build the condition for filtering
  // pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond (new pcl::ConditionOr<pcl::PointNormal> () );
  // pcl::ConditionAnd<pcl::PointNormal>::Ptr range_cond (new pcl::ConditionAnd<pcl::PointNormal> () );
  // range_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (
  //                              new pcl::FieldComparison<pcl::PointNormal> ("curvature", pcl::ComparisonOps::GT, threshold))
  //                            );

  // Build the filter
  // pcl::ConditionalRemoval<pcl::PointNormal> condrem;
  // condrem.setCondition (range_cond);
  // condrem.setInputCloud (doncloud);

  // pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered (new pcl::PointCloud<pcl::PointNormal>);

  // Apply filter
  // condrem.filter (*doncloud_filtered);

  // for(int i; i<doncloud->size();++i){
  //   if( (doncloud->points[i].curvature <= threshold) &&
  //       (normals_small_scale->points[i].normal_z >= threshold_small_z) &&
  //       (normals_large_scale->points[i].normal_z >= threshold_large_z) )
  //       {
  //         // this is the road
  //       }
  //   else
  //       doncloud_filtered->push_back(doncloud->points[i]);
  // }


  Eigen::MatrixXf Xout;
  Xout.resize(doncloud->size(),3);

  int j=0;
  for(int i=0; i<doncloud->size();++i){
    if( (doncloud->points[i].curvature >= threshold_curv_lb) &&
        (doncloud->points[i].curvature <= threshold_curv_ub) &&
        (normals_small_scale->points[i].normal_z >= threshold_small_nz_lb) &&
        (normals_small_scale->points[i].normal_z <= threshold_small_nz_ub) &&
        (normals_large_scale->points[i].normal_z >= threshold_large_nz_lb) &&
        (normals_large_scale->points[i].normal_z <= threshold_large_nz_ub) )
        {
          Xout(j,0)=doncloud->points[i].x;
          Xout(j,1)=doncloud->points[i].y;
          Xout(j,2)=doncloud->points[i].z;

          // Xout(j,3)=normals_small_scale->points[i].normal_x;
          // Xout(j,4)=normals_small_scale->points[i].normal_y;
          // Xout(j,5)=normals_small_scale->points[i].normal_z;
          //
          // Xout(j,6)=normals_large_scale->points[i].normal_x;
          // Xout(j,7)=normals_large_scale->points[i].normal_y;
          // Xout(j,8)=normals_large_scale->points[i].normal_z;
          ++j;
        }
  }


  Xout=Xout.block(0,0,j,3);


  ret.push_back(std::make_pair("Xout",Xout));

  return ret;

}
