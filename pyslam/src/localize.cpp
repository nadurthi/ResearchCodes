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


Localize::Localize(std::string opt) : kdtree(false),octree(1.0f){
        options=json::parse(opt);
        std::cout << "dmax = "<<static_cast<float>( options["Localize"]["dmax"] ) << std::endl;
        std::cout << "sig0 = "<<static_cast<float>( options["Localize"]["sig0"] ) << std::endl;


        if (options["Localize"]["octree"]["enable"])
                octree=pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(options["Localize"]["octree"]["resolution"]);

}
void Localize::setOptions(std::string opt){
        options=json::parse(opt);
}

void Localize::setMapX(const Eigen::Ref<const Eigen::MatrixXf> &MapX){
        mapX.reset(new pcl::PointCloud<pcl::PointXYZ>(MapX.rows(),1));
        // mapX->points.resize(MapX.rows());

        int i=0;
        for(auto& p: *mapX) {
                p.x = MapX(i,0);
                p.y = MapX(i,1);
                p.z = MapX(i,2);
                ++i;
        }
        kdtree.setInputCloud (mapX);
        std::cout << "Built KDtree " << std::endl;

        if (options["Localize"]["octree"]["enable"]) {
                octree.setInputCloud (mapX);
                octree.addPointsFromInputCloud ();
                std::cout << "Built Octtree " << std::endl;
        }

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


float Localize::getNNsqrddist2Map(pcl::PointXYZ searchPoint,float dmax){
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        float d=-1;
        float dmaxsq = std::pow(dmax,2);
        float res = options["Localize"]["octree"]["resolution"];
        if (options["Localize"]["octree"]["enable"]) {
                if(options["Localize"]["octree"]["UseApprxNN"]) {
                        int pointIdxRadiusSearch;
                        float pointRadiusSquaredDistance;
                        octree.approxNearestSearch(searchPoint,pointIdxRadiusSearch,pointRadiusSquaredDistance);
                        if (pointRadiusSquaredDistance<=dmaxsq)
                                d = pointRadiusSquaredDistance;

                }
                else if(options["Localize"]["octree"]["UseRadiusSearch"]) {
                        if ( octree.radiusSearch (searchPoint, dmax, pointIdxRadiusSearch, pointRadiusSquaredDistance,1) > 0 )
                        {
                                d=pointRadiusSquaredDistance[0];
                        }
                }
                else if(octree.isVoxelOccupiedAtPoint(searchPoint)) {
                        d=std::pow(res/2,2);
                }
        }
        else{
                if ( kdtree.radiusSearch (searchPoint, dmax, pointIdxRadiusSearch, pointRadiusSquaredDistance,1) > 0 )
                {
                        d=pointRadiusSquaredDistance[0];
                }
        }
        if(d==-1)
                d=dmaxsq;

        return d;

}



/**
   Xposes = [x,y,z, phi,xi,zi]
   phi is about z
   xi is about y
   zi is about x

 **/
std::vector<std::pair<std::string,Eigen::MatrixXf> >
Localize::computeLikelihood(const Eigen::Ref<const Eigen::MatrixXf> &Xposes,
                            const Eigen::Ref<const Eigen::MatrixXf> &Xmeas){

        std::vector<std::pair<std::string,Eigen::MatrixXf> > ret;
        // measX->points.resize(Xmeas.rows());

        float dmax = static_cast<float>( options["dmax"] );
        float sig0=static_cast<float>( options["sig0"] );


        float sig2 = std::pow(sig0*std::sqrt(Xmeas.rows()),2);
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
                s.resize(Xmeas.rows());
                for(std::size_t i=0; i<Xmeas.rows(); ++i) {
                        Eigen::Vector3f xm=Xmeas.row(i).head(3);
                        Eigen::Vector3f xt=R*xm+t;
                        s[i]=getNNsqrddist2Map(pcl::PointXYZ(xt(0),xt(1),xt(2)),dmax);
                }

                likelihoods[j]=std::accumulate(s.begin(), s.end(), 0)/sig2;

        }
        Eigen::MatrixXf Lk(1,Xposes.rows());
        for(std::size_t j=0; j<Xposes.rows(); ++j) {
                Lk(j)=likelihoods[j];
        }
        ret.push_back(std::make_pair("likelihood",Lk));







        return ret;
}
