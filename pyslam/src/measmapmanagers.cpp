#include "measmapmanagers.h"


MapLocalizer::MapLocalizer(std::string opt ) : bm(opt), kdtree(false),octree(1.0f){
        options=json::parse(opt);

        Vel.row(0,0)=0;
        Vel.row(0,1)=0;
        Vel.row(0,1)=0;

        AngVel.row(0,0)=0;
        AngVel.row(0,1)=0;
        AngVel.row(0,1)=0;

        XseqPos.row(0,0)=0;
        XseqPos.row(0,1)=0;
        XseqPos.row(0,1)=0;

        i1Hi_seq.clear();

        gHk.clear();
        gHk.push_back(Eigen::Matrix4f::Identity());

        setOptions(opt);


}
void MapLocalizer::setOptions(std::string opt){
        options=json::parse(opt);
        bm.setOptions(opt);

        octree=pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(options["Localize"]["octree"]["resolution"]);

        gicp.setMaxCorrespondenceDistance(options["mapfit"]["gicp"]["setMaxCorrespondenceDistance"]);
        gicp.setMaximumIterations(options["mapfit"]["gicp"]["setMaximumIterations"]);
        gicp.setMaximumOptimizerIterations(options["mapfit"]["gicp"]["setMaximumOptimizerIterations"]);
        gicp.setRANSACIterations(options["mapfit"]["gicp"]["setRANSACIterations"]);
        gicp.setRANSACOutlierRejectionThreshold(options["mapfit"]["gicp"]["setRANSACOutlierRejectionThreshold"]);
        gicp.setTransformationEpsilon(options["mapfit"]["gicp"]["setTransformationEpsilon"]);
        if(options["mapfit"]["gicp"]["setUseReciprocalCorrespondences"]==1)
                gicp.setUseReciprocalCorrespondences(1); //0.1
        else
                gicp.setUseReciprocalCorrespondences(0); //0.1

}

void MapLocalizer::resetH(){
        Vel = MatrixX3f({{0,0,0}});
        AngVel = MatrixX3f({{0,0,0}});
        XseqPos= MatrixX3f({{0,0,0}});

        i1Hi_seq.clear();
        gHk.clear();
        gHk.push_back(Eigen::Matrix4f::Identity());
}

//-----------------Setters------------------
void MapLocalizer::addMeas(const Eigen::Ref<const Eigen::MatrixXf> &X,float t){
        pcl::PointCloud<pcl::PointXYZ>::Ptr C1(new pcl::PointCloud<pcl::PointXYZ>(X.rows(),1));
        eigen2pcl(X,C1,false);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>());
        pcl_filter_cloud(C1,cloud_filtered,options["MeasMenaager"]["meas"]["downsample"]["resolution"])


        meas.push_back(cloud_filtered);
        T.push_back(t);
        ++tk;
}
void MapLocalizer::addMap(const Eigen::Ref<const Eigen::MatrixXf> &X){
        map.reset(new pcl::PointCloud<pcl::PointXYZ>(X.rows(),1));
        eigen2pcl(X,map,false);

        pcl::PointCloud<pcl::PointXYZ>::Ptr map_filtered (new pcl::PointCloud<pcl::PointXYZ>());
        pcl_filter_cloud(map,map_filtered,options["MapManager"]["map"]["downsample"]["resolution"])


        map= map_filtered;

        kdtree.setInputCloud (map);
        std::cout << "Built KDtree " << std::endl;


        octree.setInputCloud (map);
        octree.addPointsFromInputCloud ();
        std::cout << "Built Octtree " << std::endl;


        gicp.setInputTarget(map);

}
void MapLocalizer::addMap2D(const Eigen::Ref<const MatrixX2f> &X){
        map2D.reset(new pcl::PointCloud<pcl::PointXYZ>(X.rows(),1));
        int i=0;
        for(auto& p: *map2D) {
                p.x = X(i,0);
                p.y = X(i,1);
                p.z = 0;
                ++i;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr map2D_filtered (new pcl::PointCloud<pcl::PointXYZ>());
        pcl_filter_cloud(map2D,map2D_filtered,options["MapManager"]["map2D"]["downsample"]["resolution"]);

        map2D=map2D_filtered;

        bm.computeHlevels(X);
}

void MapLocalizer::setHlevels(){
        bm.computeHlevels(X);
}

void MapLocalizer::setgHk(int tk, Eigen::Matrix3f gHs ){
        gHk[tk] = gHs;
}
void MapLocalizer::setLookUpDist(){
        double min_x,min_y,min_z,max_x,max_y,max_z;
        octree.getBoundingBox(min_x,min_y,min_z,max_x,max_y,max_z);
        float x_res=options["Localize"]["lookup"]["resolution"][0];
        float y_res=options["Localize"]["lookup"]["resolution"][1];
        float z_res=options["Localize"]["lookup"]["resolution"][2];

        float dmax = static_cast<float>( options["Localize"]["dmax"] );

        int nx = (max_x-min_x)/x_res;
        ArrayXf x_edges(nx),y_edges(ny),z_edges(nz);
        for (int i=0; i<nx; ++i)
                x_edges(i)=i*x_res;
        for (int i=0; i<ny; ++i)
                y_edges(i)=i*y_res;

        for (int i=0; i<nz; ++i)
                z_edges(i)=i*z_res;

        for (int i=0; i<nx; ++i) {
                for (int i=0; i<ny; ++i) {
                        for (int i=0; i<nz; ++i) {
                                float dsq = getNNsqrddist2Map(pcl::PointXYZ(x_edges(i),y_edges(j),z_edges(k)),dmax);
                                float d=std::sqrt(dsq);
                                Xdist(i,j,k) = static_cast<uint16_t>(1000.0*d);
                        }
                }
        }


}
void MeasManager::setRegisteredSeqH(){

        for(std::size_t i=0; i<meas.size()-1; ++i) {
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
std::vector<Eigen::Matrix4f> MapLocalizer::setSeq_gHk(){
        gHk.clear();
        gHk.push_back(Eigen::Matrix4f::Zero());
        for(std::size_t i=1; i<meas.size(); ++i) {
                if(i<gHk.size())
                        continue;
                else
                        gHk.push_back(i1Hi_seq[std::make_pair(i-1,i)]*gHk[i-1]);
        }
        return gHk;
}

void MapLocalizer::setRelStates(){

        gHk = getSeq_gHk();
        for(std::size_t i=0; i<meas.size()-1; ++i) {
                if(i+1>=Vel.rows()) {
                        auto kk=std::make_pair(i,i+1);
                        H = i1Hi_seq[kk];
                        dt = T[i+1]-T[i];
                        Vel.row(i+1)=H.block(0,3,0,3)/dt;
                        AngVel.row(i+1)=H.block(0,3,3,3).eulerAngles(2, 1, 0);
                        XseqPos.row(i+1)=gHk[i+1];
                }
        }


}
//-------------------Getters------------
Eigen::MatrixXf
MapLocalizer::getmeas_eigen(int k){
        Eigen::MatrixXf X;
        pcl2eigen(meas[k],X);
        return X;
}
pcl::PointCloud<pcl::PointXYZ>::ConstPtr
MapLocalizer::getmeas(int k){
        return meas[k];
}


float MapLocalizer::getdt(){
        int n = T.size();
        if(n>=2)
                return T[n-1]-T[n-2];
        else
                return 0;
}
Vector6f MapLocalizer::MapPcllimits(){
        pcl::PointXYZ min_pt,max_pt;
        pcl::getMinMax3D (map, min_pt, max_pt);
        Vector6f lms({min_pt.x,min_pt.y,min_pt.z,max_pt.x,max_pt.y,max_pt.z});
        return lms;
}

Eigen::MatrixXf
MapLocalizer::getmaplocal_eigen(Eigen::Vector3f lb,Eigen::Vector3f ub){
        pcl::CropBox<pcl::PointXYZ> boxFilter;
        boxFilter.setMin(Eigen::Vector4f(lb(0), lb(1), lb(2), 1.0));
        boxFilter.setMax(Eigen::Vector4f(ub(0), ub(1), ub(2), 1.0));
        boxFilter.setInputCloud(map);
        pcl::PointCloud<pcl::PointXYZ>::Ptr bodyFiltered;
        boxFilter.filter(*bodyFiltered);

        Eigen::MatrixXf X;
        pcl2eigen(bodyFiltered,X);
        return X;
}
pcl::PointCloud<pcl::PointXYZ>::ConstPtr
MapLocalizer::getmaplocal(Eigen::Vector3f lb,Eigen::Vector3f ub){
        pcl::CropBox<pcl::PointXYZ> boxFilter;
        boxFilter.setMin(Eigen::Vector4f(lb(0), lb(1), lb(2), 1.0));
        boxFilter.setMax(Eigen::Vector4f(ub(0), ub(1), ub(2), 1.0));
        boxFilter.setInputCloud(map);
        pcl::PointCloud<pcl::PointXYZ>::Ptr bodyFiltered;
        boxFilter.filter(*bodyFiltered);

        return bodyFiltered;
}

Eigen::MatrixXf MapLocalizer::getmap_eigen(){
        Eigen::MatrixXf X;
        pcl2eigen(map,X);
        return X;
}
pcl::PointCloud<pcl::PointXYZ>::ConstPtr MapLocalizer::getmap(){
        return map;
}

Eigen::MatrixXf MapLocalizer::getmap2D_eigen(){
        Eigen::MatrixXf X;
        pcl2eigen(map2D,X);
        return X;
}
pcl::PointCloud<pcl::PointXYZ>::ConstPtr MapLocalizer::getmap2D(){
        return map2D;
}
MatrixX3f getvelocities(){
        return Vel;
}

MatrixX3f getpositions(){
        return XseqPos;
}
MatrixX3f getangularvelocities(){
        return AngVel;
}

VectorXf MapLocalizer::getLikelihoods(const Eigen::Ref<const Eigen::MatrixXf> &Xposes,int tk){
        if(options["Localize"]["likelihoodsearchmethod"]==std::string("octree"))
                return computeLikelihood(octree,Xposes,meas.at(tk),optionsD["Localize"]["dmax"],options["Localize"]["sig0"]);

        else if(options["Localize"]["likelihoodsearchmethod"]==std::string("lookup"))
                return computeLikelihood_lookup(Xdist,Xposes,meas.at(tk));

}


std::vector<Eigen::Matrix4f> MapLocalizer::getSeq_gHk(){
        return gHk;
}
std::vector<Eigen::Matrix4f> MapLocalizer::getsetSeq_gHk(int t0,int tf,int tk, Eigen::Matrix4f gHk){
        t2 = std::min(tf,static_cast<int>(gHk.size())-1);
        t1 = std::max(t0,0);
        auto gHkset = getSeq_gHk();
        gHkset.at(tk) = gHk;
        for(std::size_t k=tk+1; k<=t2; ++k)
                gHkset[k]=i1Hi_seq[std::make_pair(k-1,k)]*gHkset[k-1];
        for(std::size_t k=tk-1; k>=t1; --k) {
                if(k>=0) {
                        H = i1Hi_seq[std::make_pair(k,k+1)].inverse();
                        gHkset[k]=H*gHkset[k+1];
                }
        }
        std::vector<Eigen::Matrix4f>
        return gHkset;
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr MapLocalizer::getalignSeqMeas(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim){
        auto gHkset=getsetSeq_gHk(t0,tf,tk, gHk);
        pcl::PointCloud<pcl::PointXYZ> mergedpcl;
        for(std::size_t k=t0; k<=tf; ++k) {
                pclptr = getmeas(k);
                pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
                pcl::transformPointCloud (*pclptr, *transformed_cloud, gHkset[k]);

                mergedpcl=mergedpcl+*transformed_cloud;
        }
        if(dim==2) {
                for(auto &p:mergedpcl) {
                        p.z=0;
                }
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr outputcld;
        pcl_filter_cloud(&mergedpcl,outputcld,res);
        return outputcld;
}

MatrixX3f MapLocalizer::getalignSeqMeas_eigen(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim){

        outputcld = getalignSeqMeas(t0,tf,tk,gHk,res,dim);

        MatrixX3f X;
        pcl2eigen(outputcld,X);
        return X;
}


//-----------------Aligners-------------------------

BMatchAndCorrH
MapLocalizer::BMatchseq(int t0,int tf,int tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHk,bool gicp=true){
        std::vector<float> res(bm.dxMatch(0),bm.dxMatch(1),1);

        pcl::PointCloud<pcl::PointXYZ>::ConstPtr Xsrcpcl= getalignSeqMeas(t0,tf,tk, gHk,res,3);
        MatrixX3f Xsrc;
        pcl2eigen(Xsrcpcl,Xsrc);
        MatrixX2f Xsrc2D= Xsrc.block(Eigen::all,Eigen::seq(0,1));

        Eigen::Matrix3f H12;
        H12.block(0,0,2,2) = gHk.block(0,0,2,2);
        H12(0,2) = gHk(0,3);
        H12(1,2) = gHk(1,3);

        BMatchAndCorrH solret;
        solret.sols =bm.getmatch(Xsrc2D,H12);

        solret.gHkcorr = gHk;
        solret.gHkcorr.block(0,0,2,2) = sols[0].H.block(0,0,2,2);
        solret.gHkcorr(0,3) = sols[0].H(0,2);
        solret.gHkcorr(1,3) = sols[0].H(1,2);

        if(gicp==true)
                solret.gHkcorr = gicp_correction(Xsrcpcl, solret.gHkcorr);

        return solret;

}



// gHk takes k-frame local to gloal inertial frame
Eigen::Matrix4f
MapLocalizer::gicp_correction(pcl::PointCloud<pcl::PointXYZ>::ConstPtr Xsrcpcl, const Eigen::Ref<const Eigen :: Matrix4f>&gHk_est){


        pcl::PointCloud<pcl::PointXYZ>::Ptr resultXgicp(new pcl::PointCloud<pcl::PointXYZ>(Xsrcpcl.size(),1));
        pcl::transformPointCloud (*Xsrcpcl, *resultXgicp, gHk_est);
        gicp.setInputSource(resultXgicp);

        gicp.align(*resultXgicp);
        auto H_gicp = gicp.getFinalTransformation();

        return H_gicp;
}
