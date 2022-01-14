#include "donseg.h"
#include <omp.h>



std::vector<std::pair<std::string,Eigen::MatrixXf>>
donsegmentation(std::string opt,const Eigen::Ref<const Eigen::MatrixXf> &Xmap){
  std::vector<std::pair<std::string,Eigen::MatrixXf>> ret;
  auto options=json::parse(opt);
  ///The smallest scale to use in the DoN filter.
  double scale1=options["DON"]["scale1"];

  ///The largest scale to use in the DoN filter.
  double scale2=options["DON"]["scale2"];

  ///The minimum DoN magnitude to threshold by
  double threshold=options["DON"]["threshold"];

  ///segment scene into clusters with given distance tolerance using euclidean clustering
  double segradius=options["DON"]["segradius"];

  pcl::PointCloud<pcl::PointXYZ>::Ptr mapX;
  mapX.reset(new pcl::PointCloud<pcl::PointXYZ>(Xmap.rows(),1));
  // mapX->points.resize(MapX.rows());

  int i=0;
  for(auto& p: *mapX){
    p.x = Xmap(i,0);
    p.y = Xmap(i,1);
    p.z = Xmap(i,2);
    ++i;
  }
  pcl::search::Search<pcl::PointXYZ>::Ptr tree;
  tree.reset (new pcl::search::KdTree<pcl::PointXYZ> (false));
  tree->setInputCloud (mapX);

  if (scale1 >= scale2)
  {
    std::cerr << "Error: Large scale must be > small scale!" << std::endl;
    exit (EXIT_FAILURE);
  }

  // Compute normals using both small and large scales at each point
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> ne;
  ne.setInputCloud (mapX);
  ne.setSearchMethod (tree);
  /**
   * NOTE: setting viewpoint is very important, so that we can ensure
   * normals are all pointed in the same direction!
   */
  ne.setViewPoint (std::numeric_limits<float>::max (), std::numeric_limits<float>::max (), std::numeric_limits<float>::max ());

  // calculate normals with the small scale
  pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale (new pcl::PointCloud<pcl::PointNormal>);
  ne.setRadiusSearch (scale1);
  ne.compute (*normals_small_scale);


  // calculate normals with the large scale
  pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale (new pcl::PointCloud<pcl::PointNormal>);
  ne.setRadiusSearch (scale2);
  ne.compute (*normals_large_scale);

  // Create output cloud for DoN results
  pcl::PointCloud<pcl::PointNormal>::Ptr doncloud (new pcl::PointCloud<pcl::PointNormal>);
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

  // Filter by magnitude
  // Build the condition for filtering
  pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond (new pcl::ConditionOr<pcl::PointNormal> () );
  range_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (
                               new pcl::FieldComparison<pcl::PointNormal> ("curvature", pcl::ComparisonOps::GT, threshold))
                             );
  // Build the filter
  pcl::ConditionalRemoval<pcl::PointNormal> condrem;
  condrem.setCondition (range_cond);
  condrem.setInputCloud (doncloud);

  pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered (new pcl::PointCloud<pcl::PointNormal>);

  // Apply filter
  condrem.filter (*doncloud_filtered);


  doncloud = doncloud_filtered;

  Eigen::MatrixXf Xout;
  Xout.resize(doncloud->size(),3);

  i=0;
  for(auto& p: *doncloud){
    Xout(i,0)=p.x;
    Xout(i,1)=p.y;
    Xout(i,2)=p.z;
    ++i;
  }

  ret.push_back(std::make_pair("Xout",Xout));

  return ret;

}
