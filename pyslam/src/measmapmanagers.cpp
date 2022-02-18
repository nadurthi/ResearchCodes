#include "measmapmanagers.h"




MapLocalizer::MapLocalizer(std::string opt ) : bm(opt){
        options=json::parse(opt);
        kdtree.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>(false));
        octree.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(1.0f));

        resetH();

        setOptions(opt);


}
void MapLocalizer::setOptions(std::string opt){
        options=json::parse(opt);
        bm.setOptions(opt);

        octree.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(options["Localize"]["octree"]["resolution"]));


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

        gicpseq.setMaxCorrespondenceDistance(options["seqfit"]["gicp"]["setMaxCorrespondenceDistance"]);
        gicpseq.setMaximumIterations(options["seqfit"]["gicp"]["setMaximumIterations"]);
        gicpseq.setMaximumOptimizerIterations(options["seqfit"]["gicp"]["setMaximumOptimizerIterations"]);
        gicpseq.setRANSACIterations(options["seqfit"]["gicp"]["setRANSACIterations"]);
        gicpseq.setRANSACOutlierRejectionThreshold(options["seqfit"]["gicp"]["setRANSACOutlierRejectionThreshold"]);
        gicpseq.setTransformationEpsilon(options["seqfit"]["gicp"]["setTransformationEpsilon"]);
        if(options["seqfit"]["gicp"]["setUseReciprocalCorrespondences"]==1)
                gicpseq.setUseReciprocalCorrespondences(1); //0.1
        else
                gicpseq.setUseReciprocalCorrespondences(0); //0.1

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
void MapLocalizer::addMeas(const Eigen::Ref<const Eigen::MatrixXf> &X,const Eigen::Ref<const Eigen::MatrixXf> &Xnoroad,float t){
        pcl::PointCloud<pcl::PointXYZ>::Ptr C1(new pcl::PointCloud<pcl::PointXYZ>());
        eigen2pcl(X,C1,false);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>());
        pcl_filter_cloud(C1,cloud_filtered,options["MeasMenaager"]["meas"]["downsample"]["resolution"]);

        std::sort(cloud_filtered->points.begin(),cloud_filtered->points.end(),[](const pcl::PointXYZ& a,const pcl::PointXYZ& b) -> bool {
                return a.z<b.z;
        });

        meas.push_back(cloud_filtered);

        //2D
        pcl::PointCloud<pcl::PointXYZ>::Ptr C2(new pcl::PointCloud<pcl::PointXYZ>());
        eigen2pcl(Xnoroad,C2,false);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2D (new pcl::PointCloud<pcl::PointXYZ>());
        pcl_filter_cloud(C2,cloud_filtered2D,options["MeasMenaager"]["meas2D"]["downsample"]["resolution"]);

        std::sort(cloud_filtered2D->points.begin(),cloud_filtered2D->points.end(),[](const pcl::PointXYZ& a,const pcl::PointXYZ& b) -> bool {
                return a.z<b.z;
        });

        meas_noroad.push_back(cloud_filtered2D);


        T.push_back(t);
        ++tk;
}


void MapLocalizer::addMap(const Eigen::Ref<const Eigen::MatrixXf> &X){
        map.reset(new pcl::PointCloud<pcl::PointXYZ>());
        eigen2pcl(X,map,false);

        pcl::PointCloud<pcl::PointXYZ>::Ptr map_filtered (new pcl::PointCloud<pcl::PointXYZ>());
        pcl_filter_cloud(map,map_filtered,options["MapManager"]["map"]["downsample"]["resolution"]);

        std::cout<<"AddMap-------" << std::endl;
        std::cout<<"X = "<< X.rows() << std::endl;
        std::cout<<"map = "<< map->size() << std::endl;
        std::cout<<"map_filtered = "<< map_filtered->size() << std::endl;

        map= map_filtered;

        kdtree->setInputCloud (map);
        std::cout << "Built KDtree " << std::endl;


        octree->setInputCloud (map);
        octree->addPointsFromInputCloud ();
        std::cout << "Built Octtree " << std::endl;




}
void MapLocalizer::addMap2D(const Eigen::Ref<const Eigen::MatrixXf> &X){
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
        Eigen::MatrixXf X1;
        pcl2eigen(map2D,X1);

        std::cout<<"addMap2D-------" << std::endl;
        std::cout<<"X = "<< X.rows() << std::endl;
        std::cout<<"map2D = "<< map2D->size() << std::endl;
        std::cout<<"map2D_filtered = "<< map2D_filtered->size() << std::endl;
        MatrixX2f X2 =X1(Eigen::all,Eigen::seq(0,1));
        bm.computeHlevels(X2);
        std::cout<<"completed bin compute levels" << std::endl;
}




void MapLocalizer::setgHk(int tk, Eigen::Matrix4f gHs ){
        gHk[tk] = gHs;
}
void MapLocalizer::setLookUpDist(std::string filename){
        boost::filesystem::path p1(filename);


        float min_x,min_y,min_z,max_x,max_y,max_z;
        // octree->getBoundingBox(min_x,min_y,min_z,max_x,max_y,max_z);
        auto mnmx = MapPcllimits();
        min_x=mnmx(0); min_y=mnmx(1); min_z=mnmx(2); max_x=mnmx(3); max_y=mnmx(4); max_z=mnmx(5);

        Xdist_min.push_back(min_x);
        Xdist_min.push_back(min_y);
        Xdist_min.push_back(min_z);

        float x_res=options["Localize"]["lookup"]["resolution"][0];
        float y_res=options["Localize"]["lookup"]["resolution"][1];
        float z_res=options["Localize"]["lookup"]["resolution"][2];

        float dmax = static_cast<float>( options["Localize"]["dmax"] );

        int nx = (max_x-min_x)/x_res;
        int ny = (max_y-min_y)/y_res;
        int nz = (max_z-min_z)/z_res;

        std::cout<<"nx = " << nx <<std::endl;
        std::cout<<"ny = " << ny <<std::endl;
        std::cout<<"nz = " << nz <<std::endl;

        std::cout<<"x_res = " << x_res <<std::endl;
        std::cout<<"y_res = " << y_res <<std::endl;
        std::cout<<"z_res = " << z_res <<std::endl;

        std::cout<<"min_x,max_x = " << min_x <<", "<< max_x <<std::endl;
        std::cout<<"min_y,max_y = " << min_y <<", "<< max_y <<std::endl;
        std::cout<<"min_z,max_z = " << min_z <<", "<< max_z <<std::endl;

        std::cout<<"Xdist_min = "<<Xdist_min[0]<<" "<<Xdist_min[1]<<" "<<Xdist_min[2]<<" "<<std::endl;

        Eigen::ArrayXf x_edges(nx),y_edges(ny),z_edges(nz);
        for (int i=0; i<nx; ++i)
                x_edges(i)=min_x+i*x_res;

        for (int i=0; i<ny; ++i)
                y_edges(i)=min_y+i*y_res;

        for (int i=0; i<nz; ++i) {
                z_edges(i)=min_z+i*z_res;
        }

        Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic> Xidx;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Xd;


        if ( boost::filesystem::exists( "idx_"+filename ) )
        {
                std::cout<<"loading setLookUpDist" << std::endl;

                Eigen::read_binary(("idx_"+filename).c_str(), Xidx);
                Eigen::read_binary(("value_"+filename).c_str(), Xd);
                std::cout<<"read setLookUpDist" << std::endl;
                uint32_t N=Xidx.rows();


                for (uint32_t cnt=0; cnt<N; ++cnt) {
                        uint16_t i=Xidx(cnt,0);
                        uint16_t j=Xidx(cnt,1);
                        uint16_t k=Xidx(cnt,2);
                        Xdist[i][j][k] = Xd(cnt);
                }

                int cnt =1000;
                uint16_t i=Xidx(cnt,0);
                uint16_t j=Xidx(cnt,1);
                uint16_t k=Xidx(cnt,2);
                pcl::PointXYZ pt(x_edges(i),y_edges(j),z_edges(k));
                float dsq = getNNsqrddist2Map(octree,pt,dmax);
                float d=std::sqrt(dsq);
                std::cout<< "(dsaved,dcomp) = " << Xdist[i][j][k] <<", " << d << std::endl;
        }
        else{
                std::cout<<"creating and saving setLookUpDist" << std::endl;
                std::atomic<int> acnt{0};
                #pragma omp parallel for num_threads(6)
                for (uint16_t i=0; i<nx; ++i) {
                        // std::cout << "(i,nx),(j,ny) = " << i << " " << nx <<", "<< j << " " << ny << std::endl;
                        std::cout << "(i,nx), = " << i << " " << nx << std::endl;
                        for (uint16_t j=0; j<ny; ++j) {
                                std::vector<float> v;
                                v.resize(nz);

                                for (uint16_t k=0; k<nz; ++k) {
                                        pcl::PointXYZ pt(x_edges(i),y_edges(j),z_edges(k));
                                        float dsq = getNNsqrddist2Map(octree,pt,dmax);
                                        float d=std::sqrt(dsq);
                                        v[k]=d;
                                }

                                for(uint16_t k=0; k<nz; ++k) {
                                        if (v[k]<dmax) {
                                                Xdist[i][j][k] = v[k];
                                                acnt++;
                                        }
                                }
                        }
                }

                int cnt;
                cnt = acnt;

                Xidx=Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(cnt,3);
                Xd=Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Zero(cnt,1);

                cnt=0;
                for (auto& e1 : Xdist)
                {
                        for (auto& e2 : e1.second) {
                                for (auto& e3 : e2.second) {
                                        Xidx(cnt,0)= e1.first;
                                        Xidx(cnt,1)= e2.first;
                                        Xidx(cnt,2)= e3.first;
                                        Xd(cnt,0) = e3.second;
                                        cnt++;
                                }
                        }
                }

                Eigen::write_binary(("idx_"+filename).c_str(), Xidx);
                Eigen::write_binary(("value_"+filename).c_str(), Xd);

                // phmap::BinaryOutputArchive ar_out("dump1.bin");
                // Xdist.phmap_dump(ar_out);
        }
        std::cout<<"completed setLookUpDist" << std::endl;


}
void MapLocalizer::setRegisteredSeqH(){

        for(std::size_t i=0; i<meas.size()-1; ++i) {
                if(i1Hi_seq.find(i)==i1Hi_seq.end()) {
                        if(i1Hi_seq[i].find(i+1)==i1Hi_seq[i].end()) {
                                gicpseq.setInputSource(meas[i]);
                                gicpseq.setInputTarget(meas[i+1]);
                                pcl::PointCloud<pcl::PointXYZ>::Ptr resultXgicp(new pcl::PointCloud<pcl::PointXYZ>());
                                gicpseq.align(*resultXgicp);
                                auto H_gicp = gicpseq.getFinalTransformation();
                                i1Hi_seq[i][i+1]=H_gicp;
                        }
                }
        }

        setSeq_gHk();

}
std::vector<Eigen::Matrix4f> MapLocalizer::setSeq_gHk(){
        gHk.clear();
        gHk.push_back(Eigen::Matrix4f::Identity());
        for(std::size_t i=1; i<meas.size(); ++i) {
                if(i<gHk.size())
                        continue;
                else
                        gHk.push_back(i1Hi_seq[i-1][i]*gHk[i-1]);
        }
        return gHk;
}

void MapLocalizer::setRelStates(){

        gHk = getSeq_gHk();
        for(std::size_t i=0; i<meas.size()-1; ++i) {

                Eigen::Matrix4f H = i1Hi_seq[i][i+1];
                auto dt = T[i+1]-T[i];
                Vel.row(i+1)=H.col(3)(Eigen::seq(0,2))/dt;
                Eigen::Matrix3f R = H.block(0,0,3,3);
                Eigen::Vector3f vv = R.eulerAngles(2, 1, 0);
                AngVel.row(i+1)=vv;
                XseqPos.row(i+1)=gHk[i+1].col(3)(Eigen::seq(0,2));

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
pcl::PointCloud<pcl::PointXYZ>::ConstPtr
MapLocalizer::getmeas_noroad(int k){
        return meas_noroad[k];
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
        pcl::getMinMax3D (*map, min_pt, max_pt);
        Vector6f lms({min_pt.x,min_pt.y,min_pt.z,max_pt.x,max_pt.y,max_pt.z});
        return lms;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
MapLocalizer::getmaplocal(Eigen::Vector3f lb,Eigen::Vector3f ub){
        pcl::CropBox<pcl::PointXYZ> boxFilter;
        boxFilter.setMin(Eigen::Vector4f(lb(0), lb(1), lb(2), 1.0));
        boxFilter.setMax(Eigen::Vector4f(ub(0), ub(1), ub(2), 1.0));
        boxFilter.setInputCloud(map);
        pcl::PointCloud<pcl::PointXYZ>::Ptr bodyFiltered;
        boxFilter.filter(*bodyFiltered);

        return bodyFiltered;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr
MapLocalizer::getmaplocal(pcl::PointXYZ min_pt,pcl::PointXYZ max_pt){
        Eigen::Vector3f lb({min_pt.x,min_pt.y,min_pt.z});
        Eigen::Vector3f ub({max_pt.x,max_pt.y,max_pt.z});
        return getmaplocal(lb,ub);
}


Eigen::MatrixXf
MapLocalizer::getmaplocal_eigen(Eigen::Vector3f lb,Eigen::Vector3f ub){
        auto bodyFiltered=getmaplocal(lb,ub);
        Eigen::MatrixXf X;
        pcl2eigen(bodyFiltered,X);
        return X;
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
MatrixX3f MapLocalizer::getvelocities(){
        return Vel;
}

MatrixX3f MapLocalizer::getpositions(){
        return XseqPos;
}
MatrixX3f MapLocalizer::getangularvelocities(){
        return AngVel;
}

Eigen::VectorXf MapLocalizer::getLikelihoods(const Eigen::Ref<const Eigen::MatrixXf> &Xposes,int tk){
        if(options["Localize"]["likelihoodsearchmethod"]==std::string("octree"))
                return computeLikelihood(octree,Xposes,meas.at(tk),options["Localize"]["dmax"],options["Localize"]["sig0"]);

        else if(options["Localize"]["likelihoodsearchmethod"]==std::string("lookup")) {
                return computeLikelihood_lookup(Xdist,options["Localize"]["lookup"]["resolution"],Xdist_min,
                                                Xposes,meas.at(tk),options["Localize"]["dmax"],options["Localize"]["sig0"]);
        }

}


std::vector<Eigen::Matrix4f> MapLocalizer::getSeq_gHk(){
        return gHk;
}
std::vector<Eigen::Matrix4f> MapLocalizer::getsetSeq_gHk(int t0,int tf,int tk, Eigen::Matrix4f gHk){
        auto t2 = std::min(tf,static_cast<int>(gHk.size())-1);
        auto t1 = std::max(t0,0);
        auto gHkset = getSeq_gHk();
        gHkset.at(tk) = gHk;
        for(std::size_t k=tk+1; k<=t2; ++k)
                gHkset[k]=i1Hi_seq[k-1][k]*gHkset[k-1];
        for(std::size_t k=tk-1; k>=t1; --k) {
                if(k>=0) {
                        auto H = i1Hi_seq[k][k+1].inverse();
                        gHkset[k]=H*gHkset[k+1];
                }
        }

        return gHkset;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr MapLocalizer::getalignSeqMeas(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim){
        auto gHkset=getsetSeq_gHk(t0,tf,tk, gHk);
        pcl::PointCloud<pcl::PointXYZ> mergedpcl;
        for(std::size_t k=t0; k<=tf; ++k) {
                auto pclptr = getmeas(k);

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
        pcl::PointCloud<pcl::PointXYZ>::Ptr mergedpclptr(&mergedpcl);
        pcl_filter_cloud(mergedpclptr,outputcld,res);
        return outputcld;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr MapLocalizer::getalignSeqMeas_noroad(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim){
        auto gHkset=getsetSeq_gHk(t0,tf,tk, gHk);
        pcl::PointCloud<pcl::PointXYZ> mergedpcl;
        for(std::size_t k=t0; k<=tf; ++k) {
                auto pclptr = getmeas_noroad(k);

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
        pcl::PointCloud<pcl::PointXYZ>::Ptr mergedpclptr(&mergedpcl);
        pcl_filter_cloud(mergedpclptr,outputcld,res);
        return outputcld;
}

Eigen :: MatrixXf MapLocalizer::getalignSeqMeas_eigen(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim){

        auto outputcld = getalignSeqMeas(t0,tf,tk,gHk,res,dim);

        Eigen :: MatrixXf X;
        pcl2eigen(outputcld,X);
        return X;
}
Eigen :: MatrixXf MapLocalizer::getalignSeqMeas_noroad_eigen(int t0,int tf,int tk, Eigen::Matrix4f gHk,std::vector<float> res,int dim){
        auto outputcld = getalignSeqMeas_noroad(t0,tf,tk,gHk,res,dim);

        Eigen :: MatrixXf X;
        pcl2eigen(outputcld,X);
        return X;

}

//-----------------Aligners-------------------------

BMatchAndCorrH
MapLocalizer::BMatchseq(int t0,int tf,int tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHk,bool gicp){
        std::vector<float> res({bm.dxMatch(0),bm.dxMatch(1),1});

        pcl::PointCloud<pcl::PointXYZ>::Ptr Xsrcpcl= getalignSeqMeas_noroad(t0,tf,tk, gHk,res,2);
        Eigen::MatrixXf Xsrc;
        pcl2eigen(Xsrcpcl,Xsrc);
        MatrixX2f Xsrc2D= Xsrc(Eigen::all,Eigen::seq(0,1));

        Vector6f xpose;
        Hmat2pose(gHk,xpose);


        Eigen::Matrix3f Rz;
        Rz = Eigen::AngleAxisf(xpose(3), Eigen::Vector3f::UnitZ());
        std::cout << "Rz = " << Rz <<std::endl;


        Eigen::Matrix3f H12=Eigen::Matrix3f::Identity();
        H12.block(0,0,2,2) = Rz.block(0,0,2,2);
        H12(0,2) = xpose(0);
        H12(1,2) = xpose(1);
        std::cout << "H12 = " << H12 <<std::endl;

        BMatchAndCorrH solret;
        solret.sols =bm.getmatch(Xsrc2D,H12);


        Eigen::Matrix3f R =Eigen::Matrix3f::Identity();
        R.block(0,0,2,2)=solret.sols[0].H.block(0,0,2,2);
        Eigen::Vector3f vv2;
        vv2 = R.eulerAngles(2, 1, 0);
        xpose(3)=vv2(0);
        xpose(0)=solret.sols[0].H(0,2);
        xpose(1)=solret.sols[0].H(1,2);

        // pose2Hmat(xpose,solret.gHkcorr);


        if(gicp==true) {
                std::vector<float> res = options["mapfit"]["downsample"]["resolution"];
                pcl::PointCloud<pcl::PointXYZ>::Ptr Xsrcpcl= getalignSeqMeas(t0,tf,tk, solret.gHkcorr,res,3);
                solret.gHkcorr = gicp_correction(Xsrcpcl, solret.gHkcorr);
        }
        return solret;

}



// gHk takes k-frame local to gloal inertial frame
Eigen::Matrix4f
MapLocalizer::gicp_correction(pcl::PointCloud<pcl::PointXYZ>::Ptr Xsrcpcl, const Eigen::Ref<const Eigen :: Matrix4f>&gHk_est){



        pcl::PointCloud<pcl::PointXYZ>::Ptr resultXgicp(new pcl::PointCloud<pcl::PointXYZ>(Xsrcpcl->size(),1));
        pcl::transformPointCloud (*Xsrcpcl, *resultXgicp, gHk_est);

        pcl::PointXYZ min_pt,max_pt;
        pcl::getMinMax3D (*Xsrcpcl, min_pt, max_pt);
        int extra=10;
        min_pt.x-=extra; min_pt.y-=extra;; min_pt.z-=extra;
        min_pt.x+=extra; min_pt.y+=extra;; min_pt.z+=extra;
        auto maplocal = getmaplocal(min_pt,max_pt);

        gicp.setInputTarget(maplocal);
        gicp.setInputSource(resultXgicp);

        gicp.align(*resultXgicp);
        auto H_gicp = gicp.getFinalTransformation();

        return H_gicp*gHk_est;
}
