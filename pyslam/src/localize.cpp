#include "localize.h"
#include <omp.h>

void pose2Hmat(const Vector6f& x,Eigen::Matrix4f& H){
        H=Eigen::Matrix4f::Identity();

        Eigen::Matrix3f R;
        R = Eigen::AngleAxisf(static_cast<float>(x[3]), Eigen::Vector3f::UnitZ()) *
            Eigen::AngleAxisf(static_cast<float>(x[4]), Eigen::Vector3f::UnitY()) *
            Eigen::AngleAxisf(static_cast<float>(x[5]), Eigen::Vector3f::UnitX());
        H.topLeftCorner<3, 3>().matrix() = R; //* t.topLeftCorner<3, 3>().matrix();
        // Eigen::Vector4f T(static_cast<float>(x[0]),
        //                   static_cast<float>(x[1]),
        //                   static_cast<float>(x[2]),
        //                   1.0f);
        H(0,3) = x(0);
        H(1,3) = x(1);
        H(2,3) = x(2);

}

void Hmat2pose(const Eigen::Matrix4f& H,Vector6f& x){


        Eigen::Matrix3f R=H.block(0,0,3,3);
        Eigen::Vector3f vv =R.eulerAngles(2,1,0);

        x(0)=H(0,3);
        x(1)=H(1,3);
        x(2)=H(2,3);
        x(3)=vv(0);
        x(4)=vv(1);
        x(5)=vv(2);

}



// void Localize::setMapX( pcl::PointCloud<pcl::PointXYZ> MapX){
//         mapX.reset(MapX);
//         // mapX->points.resize(MapX.rows());
//
//         kdtree.setInputCloud (mapX);
//         icp.setInputTarget(mapX);
//         // octree.setInputCloud (mapX);
//         // octree.addPointsFromInputCloud ();
//
//         std::cout << "Built KDtree " << std::endl;
// }

float getNNsqrddist2Map(pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr mapoctree,const pcl::PointXYZ & searchPoint,float dmax){
        float dmaxsq = std::pow(dmax,2);
        int pointIdxRadiusSearch;
        float pointRadiusSquaredDistance;
        mapoctree->approxNearestSearch(searchPoint,pointIdxRadiusSearch,pointRadiusSquaredDistance);
        float d;
        if (pointRadiusSquaredDistance<=dmaxsq)
                d = pointRadiusSquaredDistance;
        else
                d=dmaxsq;

        return d;
}
float getNNsqrddist2Map(pcl::KdTree<pcl::PointXYZ>::Ptr mapkdtree,const pcl::PointXYZ & searchPoint,float dmax){
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        float dmaxsq = std::pow(dmax,2);
        float d;
        if ( mapkdtree->radiusSearch (searchPoint, dmax, pointIdxRadiusSearch, pointRadiusSquaredDistance,1) > 0 )
                d=pointRadiusSquaredDistance[0];
        else
                d=dmaxsq;

        return d;

}





/**
   Xposes = [x,y,z, phi,xi,zi]
   phi is about z
   xi is about y
   zi is about x

 **/
VectorXf
computeLikelihood(pcl::KdTree<pcl::PointXYZ>::Ptr mapkdtree,
                  const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                  pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl,float dmax,float sig0){





        float sig2 = std::pow(sig0*std::sqrt(Xmeaspcl->size()),2);
        std::cout << "sig2 = " << sig2 << std::endl;




        std::vector<float> likelihoods;
        likelihoods.resize(Xposes.rows());

        // omp_set_dynamic(0);

        //
  #pragma omp parallel for num_threads(6)
        for(std::size_t j=0; j<Xposes.rows(); ++j) {
                // std::cout << "j = " << j << std::endl;

                Vector6f xx = Xposes.row(j).head(6);

                Eigen::Matrix4f H;
                pose2Hmat(xx,H);
                Eigen::Matrix3f R=H.topLeftCorner<3, 3>().matrix();
                Eigen::Vector3f t= Eigen::Vector3f::Zero();
                t(0)=H(0,3);
                t(1)=H(1,3);
                t(2)=H(2,3);

                std::vector<float> s;
                s.resize(Xmeaspcl->size());
                for(std::size_t i=0; i<Xmeaspcl->size(); ++i) {
                        Eigen::Vector3f xm({Xmeaspcl->points[i].x,Xmeaspcl->points[i].y,Xmeaspcl->points[i].z});
                        Eigen::Vector3f xt=R*xm+t;
                        s[i]=getNNsqrddist2Map(mapkdtree,pcl::PointXYZ(xt(0),xt(1),xt(2)),dmax);
                }

                likelihoods[j]=std::accumulate(s.begin(), s.end(), 0)/sig2;

        }

        VectorXf likelihoods_eig;
        for (std::size_t i=0; i<likelihoods.size(); ++i)
                likelihoods_eig(i)=likelihoods[i];

        return likelihoods_eig;
}

VectorXf
computeLikelihood(pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr mapoctree,
                  const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                  pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl,float dmax,float sig0){





        float sig2 = std::pow(sig0*std::sqrt(Xmeaspcl->size()),2);
        std::cout << "sig2 = " << sig2 << std::endl;




        std::vector<float> likelihoods;
        likelihoods.resize(Xposes.rows());

        // omp_set_dynamic(0);

        //
  #pragma omp parallel for num_threads(6)
        for(std::size_t j=0; j<Xposes.rows(); ++j) {
                // std::cout << "j = " << j << std::endl;

                Vector6f xx = Xposes.row(j).head(6);

                Eigen::Matrix4f H;
                pose2Hmat(xx,H);
                Eigen::Matrix3f R=H.topLeftCorner<3, 3>().matrix();
                Eigen::Vector3f t= Eigen::Vector3f::Zero();
                t(0)=H(0,3);
                t(1)=H(1,3);
                t(2)=H(2,3);

                std::vector<float> s;
                s.resize(Xmeaspcl->size());
                for(std::size_t i=0; i<Xmeaspcl->size(); ++i) {
                        Eigen::Vector3f xm({Xmeaspcl->points[i].x,Xmeaspcl->points[i].y,Xmeaspcl->points[i].z});
                        Eigen::Vector3f xt=R*xm+t;
                        s[i]=getNNsqrddist2Map(mapoctree,pcl::PointXYZ(xt(0),xt(1),xt(2)),dmax);
                }

                likelihoods[j]=std::accumulate(s.begin(), s.end(), 0)/sig2;

        }


        VectorXf likelihoods_eig;
        for (std::size_t i=0; i<likelihoods.size(); ++i)
                likelihoods_eig(i)=likelihoods[i];

        return likelihoods_eig;
}

VectorXf
computeLikelihood_lookup(const std::vector<MatrixXXuint16> &Xdist, const std::vector<float>& res,
                         const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                         pcl::PointCloud<pcl::PointXYZ >::ConstPtr Xmeaspcl,float dmax,float sig0){




        float sig2 = std::pow(sig0*std::sqrt(Xmeaspcl->size()),2);
        std::cout << "sig2 = " << sig2 << std::endl;




        std::vector<float> likelihoods;
        likelihoods.resize(Xposes.rows());

        // omp_set_dynamic(0);

        //
  #pragma omp parallel for num_threads(6)
        for(std::size_t j=0; j<Xposes.rows(); ++j) {
                // std::cout << "j = " << j << std::endl;

                Vector6f xx = Xposes.row(j).head(6);

                Eigen::Matrix4f H;
                pose2Hmat(xx,H);
                Eigen::Matrix3f R=H.topLeftCorner<3, 3>().matrix();
                Eigen::Vector3f t= Eigen::Vector3f::Zero();
                t(0)=H(0,3);
                t(1)=H(1,3);
                t(2)=H(2,3);

                std::vector<float> s;
                s.resize(Xmeaspcl->size());
                for(std::size_t i=0; i<Xmeaspcl->size(); ++i) {
                        Eigen::Vector3f xm({Xmeaspcl->points[i].x,Xmeaspcl->points[i].y,Xmeaspcl->points[i].z});
                        Eigen::Vector3f xt=R*xm+t;
                        int p=int(xt(0)/res[0]);
                        int q=int(xt(1)/res[1]);
                        int r=int(xt(2)/res[2]);
                        float d = static_cast<float>(Xdist[r](p,q))/1000;
                        s[i] = std::pow(d,2);
                }

                likelihoods[j]=std::accumulate(s.begin(), s.end(), 0)/sig2;

        }


        VectorXf likelihoods_eig;
        for (std::size_t i=0; i<likelihoods.size(); ++i)
                likelihoods_eig(i)=likelihoods[i];

        return likelihoods_eig;
}
