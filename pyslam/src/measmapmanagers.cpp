#include "measmapmanagers.h"

void pose2Hmat_(const Vector6f& x,Eigen::Matrix4f& H){
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


MeasManager::MeasManager(std::string opt ){
        options=json::parse(opt);
}

void MeasManager::addMeas(const Eigen::Ref<const Eigen::MatrixXf> &X,float t){
        pcl::PointCloud<pcl::PointXYZ>::Ptr C1(new pcl::PointCloud<pcl::PointXYZ>(X.rows(),1));
        int i=0;
        for(auto& p: *C1) {
                p.x = X(i,0);
                p.y = X(i,1);
                p.z = X(i,2);
                ++i;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud (C1);
        sor.setLeafSize(options["meas"]["x_down"], options["meas"]["y_down"], options["meas"]["z_down"]);
        sor.filter (*cloud_filtered);


        meas.push_back(cloud_filtered);
        T.push_back(t);

}


float MeasManager::MeasManager::getdt(){
        int n = T.size();
        if(n>2)
                return T[n]-T[n-1];
        else
                return 0;
}


void MeasManager::resetH(){

        i1Hi_seq.clear();
        gHs.clear();
        gHs.push_back(Eigen::Matrix4f::Zero());

}


void MeasManager::computeSeqH(){
        for(int i=1; i<meas.size(); ++i) {
                if(i<gHs.size())
                        continue;
                else
                        gHs[i]=i1Hi_seq[std::make_pair(i-1,i)]*gHs[i-1];
        }
}


void MeasManager::registerSeqMeas(){

        for(int i=0; i<meas.size()-1; ++i) {
                auto kk=std::make_pair(i,i+1);
                if(i1Hi_seq.find(kk)!=i1Hi_seq.end()) {
                        gicp.setInputSource(meas[i]);
                        gicp.setInputTarget(meas[i+1]);
                        pcl::PointCloud<pcl::PointXYZ>::Ptr resultXgicp(new pcl::PointCloud<pcl::PointXYZ>);
                        gicp.align(*resultXgicp);
                        auto H_gicp = gicp.getFinalTransformation();
                        i1Hi_seq[kk]=H_gicp;
                }
        }

        computeSeqH();

}



MapManager::MapManager(std::string opt ) : octree(1){
        options=json::parse(opt);
}

void MapManager::addMap(const Eigen::Ref<const Eigen::MatrixXf> &X){
        map.reset(new pcl::PointCloud<pcl::PointXYZ>(X.rows(),1));
        int i=0;
        for(auto& p: *map) {
                p.x = X(i,0);
                p.y = X(i,1);
                p.z = X(i,2);
                ++i;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr map_filtered (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud (map);
        sor.setLeafSize(options["map"]["x_down"], options["map"]["y_down"], options["map"]["z_down"]);
        sor.filter (*map_filtered);


        map= map_filtered;

}

void MapManager::addMap2D(const Eigen::Ref<const Eigen::MatrixXf> &X){
        map.reset(new pcl::PointCloud<pcl::PointXYZ>(X.rows(),1));
        int i=0;
        for(auto& p: *map) {
                p.x = X(i,0);
                p.y = X(i,1);
                p.z = 0;
                ++i;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr map_filtered (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud (map);
        sor.setLeafSize(options["map2D"]["x_down"], options["map2D"]["y_down"], options["map2D"]["z_down"]);
        sor.filter (*map_filtered);

        map2D=map_filtered;

}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr MapManager::getmap(){
        return map;
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr MapManager::getmap2D(){
        return map2D;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr MapManager::getmaplocal(Eigen::Vector3f lb,Eigen::Vector3f ub){

        pcl::PointCloud<pcl::PointXYZ>::Ptr mapcrop;
        pcl::CropBox<pcl::PointXYZ> boxFilter;
        boxFilter.setMin(Eigen::Vector4f(lb(0), lb(1), lb(2), 1.0));
        boxFilter.setMax(Eigen::Vector4f(ub(0), ub(1), ub(2), 1.0));
        boxFilter.setInputCloud(map);
        boxFilter.filter(*mapcrop);
        return mapcrop;
}

float MapManager::getNNsqrddist2Map(pcl::PointXYZ searchPoint,float dmax){
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        float d;
        if ( kdtree.radiusSearch (searchPoint, dmax, pointIdxRadiusSearch, pointRadiusSquaredDistance,1) > 0 )
        {
                d=pointRadiusSquaredDistance[0];
                // for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
                // std::cout << "    "  <<   (*cloud)[ pointIdxRadiusSearch[i] ].x
                //           << " " << (*cloud)[ pointIdxRadiusSearch[i] ].y
                //           << " " << (*cloud)[ pointIdxRadiusSearch[i] ].z
                //           << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
        }
        else
                d=std::pow(dmax,2);

        return d;

}




/**
   Xposes = [x,y,z, phi,xi,zi]
   phi is about z
   xi is about y
   zi is about x

 **/

Eigen::VectorXf
computeLikelihood(MapManager &map,
                  const Eigen::Ref<const Eigen::MatrixXf> &Xposes, const Eigen::Ref<const Eigen::MatrixXf> &Xmeas){
        pcl::PointCloud<pcl::PointXYZ>::Ptr measX(new pcl::PointCloud<pcl::PointXYZ>(Xmeas.rows(),1));

        float dmax = static_cast<float>( map.options["dmax"] );
        float sig0=static_cast<float>( map.options["sig0"] );

        int i=0;
        for(auto& p: *measX) {
                p.x = Xmeas(i,0);
                p.y = Xmeas(i,1);
                p.z = Xmeas(i,2);
                ++i;
        }

        float sig2 = std::pow(sig0*std::sqrt(Xmeas.rows()),2);

        Eigen::VectorXf likelihoods(Xposes.rows());


        // omp_set_dynamic(0);

        //
  #pragma omp parallel for num_threads(6)
        for(int j=0; j<Xposes.rows(); ++j) {
                // std::cout << "j = " << j << std::endl;

                Vector6f xx = Xposes.row(j).head(6);

                Eigen::Matrix4f H;
                pose2Hmat_(xx,H);
                Eigen::Matrix3f R=H.topLeftCorner<3, 3>().matrix();
                Eigen::Vector3f t= Eigen::Vector3f::Zero();
                t(0)=H(0,3);
                t(1)=H(1,3);
                t(2)=H(2,3);

                std::vector<float> s;
                s.resize(Xmeas.rows());
                for(int i=0; i<Xmeas.rows(); ++i) {
                        Eigen::Vector3f xm=Xmeas.row(i).head(3);
                        Eigen::Vector3f xt=R*xm+t;
                        s[i]=map.getNNsqrddist2Map(pcl::PointXYZ(xt(0),xt(1),xt(2)),dmax);
                }

                likelihoods(j)=std::accumulate(s.begin(), s.end(), 0)/sig2;

        }

        return likelihoods;

}
