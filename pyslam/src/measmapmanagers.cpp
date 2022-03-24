#include "measmapmanagers.h"




MapLocalizer::MapLocalizer(std::string opt ) : bm(opt),plotter(opt),quitsim(false){
        options=json::parse(opt);

        resetH();

        kdtree.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>(false));
        octree.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(1.0f));



        setOptions(opt);

        timerptr=std::make_shared<timerdict>();
}

void MapLocalizer::setquitsim(){
        quitsim=true;
}

void MapLocalizer::plotsim(const Eigen::Ref<const Eigen::MatrixXf> &Xpose){
        plotter.clear();
        plotter.plotMap(map);
        // Eigen::MatrixXf dirs(Xpose.rows(),3);
        // Eigen::MatrixXf PFpoints=Xpose(Eigen::all,Eigen::seq(0,2));
        //
        // Vector3f p({1,0,0});
        // for(int i=0; i<Xpose.rows(); ++i) {
        //         auto H=pose2Hmat(Xpose.row(i));
        //         Eigen::Matrix3f R = H.block(0,0,3,3);
        //         Vector3f t = H.col(3).head(3);
        //
        //         auto drn=R*p;
        //         dirs.row(i) = drn;
        //
        // }
        // plotter.plotPFpoints(PFpoints,dirs);


        plotter.update();
}

void MapLocalizer::autoReadMeas_async(std::string folder,int k0){
        // auto autoReadMeas_future = std::async(std::launch::async, &MapLocalizer::autoReadMeas,this,folder);
        std::thread autoReadMeas_thread(&MapLocalizer::autoReadMeas,this,folder,k0);
        autoReadMeas_thread.detach();
}

void MapLocalizer::removeRoad(pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl,pcl::PointCloud<pcl::PointXYZ>::Ptr& Xpcl_noroad){
        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl_filter_cloud(Xpcl,Xpcl_filtered,options["MeasMenaager"]["measnormals"]["resolution"]);

        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> ne;
        ne.setInputCloud (Xpcl_filtered);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
        ne.setSearchMethod (tree); //kdtree
        int knn = options["MeasMenaager"]["measnormals"]["knn"];
        ne.setKSearch(knn);
        /**
         * NOTE: setting viewpoint is very important, so that we can ensure
         * normals are all pointed in the same direction!
         */
        ne.setViewPoint (std::numeric_limits<float>::max (), std::numeric_limits<float>::max (), std::numeric_limits<float>::max ());
        // ne.setViewPoint (0, 0, 0);
        // calculate normals with the small scale
        float r = options["MeasMenaager"]["measnormals"]["radius"];
        // ne.setRadiusSearch (r);
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
        ne.compute (*cloud_normals);

        float min_norm_z_thresh = options["MeasMenaager"]["measnormals"]["min_norm_z_thresh"];
        pcl::PointIndicesPtr indices = pcl::PointIndicesPtr(new pcl::PointIndices);
        for (auto i = 0; i < cloud_normals->size(); ++i)
        {
                const auto& normal = cloud_normals->points[i].getNormalVector3fMap();
                if (std::abs(normal[2]) >= min_norm_z_thresh)
                {
                        indices->indices.push_back(i);
                }
        }

        // Xpcl_noroad.reset(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::ExtractIndices<pcl::PointXYZ> filter;
        filter.setInputCloud (Xpcl_filtered);
        filter.setNegative (true);
        filter.setIndices (indices);
        filter.filter (*Xpcl_noroad);


}

void MapLocalizer::autoReadMeas(std::string folder,int k0){
        boost::filesystem::path p1(folder);

        int k=k0;
        while(1 && quitsim==false) {
                if (measQ.size()<10) {
                        std::stringstream ss;
                        ss << std::setw(6) << std::setfill('0') << k << ".bin";
                        std::string s = ss.str();
                        boost::filesystem::path f(s);
                        boost::filesystem::path pp = p1 / f;
                        if (!boost::filesystem::exists( pp)) {
                                std::cout << "******** File: \" " << pp.string() <<  " \" does not exist ******" << std::endl;
                                break;
                        }
                        std::cout << "Reading File: \" " << pp.string() <<  " \"" << std::endl;

                        Eigen::MatrixXf X;
                        int cols,rows;
                        cols=4;

                        std::ifstream in(pp.string(), std::ios::in | std::ios::binary);
                        in.seekg (0, in.end);
                        int length = in.tellg();
                        in.seekg (0, in.beg);

                        // std::cout << "file size: " << length << std::endl;
                        X.resize(4,length/16);
                        in.read( (char *) X.data(), (length/4)*sizeof(typename Eigen::MatrixXf::Scalar) );
                        in.close();

                        // std::cout<< "reading done at timestep "<<k<<std::endl;
                        // std::cout<< "X.shape = "<<X.rows() << " " << X.cols()<<std::endl;

                        Eigen::MatrixXf XT=X.transpose();
                        // Eigen::Map<Eigen::MatrixXf> M2(X.data(), length/4,4);
                        // Eigen::MatrixXf X2 = M2;
                        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl(new pcl::PointCloud<pcl::PointXYZ>(XT.rows(),1));
                        eigen2pcl(XT,Xpcl,false);

                        pcl::PointIndicesPtr indices = pcl::PointIndicesPtr(new pcl::PointIndices);
                        for (auto i = 0; i < Xpcl->size(); ++i)
                        {
                                const auto& pt = Xpcl->points[i].getVector3fMap();
                                if ((pt[0]>-200) && (pt[0]<200) && (pt[1]>-200) && (pt[1]<200) && (pt[2]>-100) && (pt[2]<100))
                                {
                                        indices->indices.push_back(i);
                                }
                        }

                        // Xpcl_noroad.reset(new pcl::PointCloud<pcl::PointXYZ>());
                        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl_flr(new pcl::PointCloud<pcl::PointXYZ>());
                        pcl::ExtractIndices<pcl::PointXYZ> filter;
                        filter.setInputCloud (Xpcl);
                        // filter.setNegative (true);
                        filter.setIndices (indices);
                        filter.filter (*Xpcl_flr);


                        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl_noroad(new pcl::PointCloud<pcl::PointXYZ>());
                        removeRoad(Xpcl_flr,Xpcl_noroad);

                        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl_filtered(new pcl::PointCloud<pcl::PointXYZ>());
                        pcl_filter_cloud(Xpcl_flr,Xpcl_filtered,options["MeasMenaager"]["meas"]["downsample"]["resolution"]);

                        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl_noroad_filtered(new pcl::PointCloud<pcl::PointXYZ>());
                        pcl_filter_cloud(Xpcl_noroad,Xpcl_noroad_filtered,options["MeasMenaager"]["meas2D"]["downsample"]["resolution"]);

                        measQmtx.lock();
                        measQ.push(Xpcl_filtered);
                        measnoroadQ.push(Xpcl_noroad_filtered);
                        measQmtx.unlock();

                        ++k;
                        // std::cout<< "added meas at timestep "<<k<<std::endl;
                }
                else{
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
        }
}

std::vector<Eigen::MatrixXf> MapLocalizer::getMeasQ_eigen(bool popit){
        Eigen::MatrixXf X,Xnr;
        std::vector<Eigen::MatrixXf> res;
        if (!measQ.empty()) {
                pcl2eigen(measQ.front(),X);
                pcl2eigen(measnoroadQ.front(),Xnr);
                if(popit) {
                        measQ.pop();
                        measnoroadQ.pop();
                }
                res.push_back(X);
                res.push_back(Xnr);
                return res;
        }
        else{
                return res;
        }
}


void MapLocalizer::cleanUp(){
        for(int i=0; i<meas.size(); ++i) {
                meas[i]->clear();
                meas_noroad[i]->clear();
        }
}
void MapLocalizer::setBMOptions(std::string opt){
        bm.setOptions(opt);
}
void MapLocalizer::setOptions_noreset(std::string opt){
        bm.setOptions(opt);
        options=json::parse(opt);
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
        Vel.clear();
        AngVel.clear();
        XseqPos.clear();

        Vel.push_back(Eigen::Vector3f::Zero());
        AngVel.push_back(Eigen::Vector3f::Zero());
        XseqPos.push_back(Eigen::Vector3f::Zero());

        i1Hi_seq.clear();

        gHk.clear();
        gHk.push_back(Eigen::Matrix4f::Identity());



        meas.clear();
        meas_noroad.clear();
        T.clear();

        map.reset();
        map2D.reset();

        octree.reset();
        kdtree.reset();

        Xdist_min.clear();

        if(bmHsols_async_future.valid()==true) {
                bmHsols_async_future.get();
        }


}
void MapLocalizer::resetsim(){
        Vel.clear();
        AngVel.clear();
        XseqPos.clear();

        Vel.push_back(Eigen::Vector3f::Zero());
        AngVel.push_back(Eigen::Vector3f::Zero());
        XseqPos.push_back(Eigen::Vector3f::Zero());

        i1Hi_seq.clear();

        gHk.clear();
        gHk.push_back(Eigen::Matrix4f::Identity());



        meas.clear();
        meas_noroad.clear();
        T.clear();

        // map.reset();
        // map2D.reset();
        //
        // octree.reset();
        // kdtree.reset();
        //
        // Xdist_min.clear();

        if(bmHsols_async_future.valid()==true) {
                bmHsols_async_future.get();
        }
        if(seqregister_future.valid()==true) {
                seqregister_future.get();
        }
        if(setrelstates_future.valid()==true) {
                setrelstates_future.get();
        }

        measQmtx.lock();
        for (int i=0; i<measQ.size(); i++) {
                std::cout << "measQ popped : "<< i << '\n';
                measQ.pop();
        }
        for (int i=0; i<measnoroadQ.size(); i++) {
                std::cout << "measnoroadQ popped : "<< i << '\n';
                measnoroadQ.pop();
        }
        measQmtx.unlock();

}
//-----------------Setters------------------
structmeas MapLocalizer::addMeas_fromQ(Eigen::Matrix4f H,float t){
        Timer TT("addMeas_fromQ",timerptr);
        T.push_back(std::round(10e4*t)/10e4);

        structmeas sm;
        sm.dt=getdt();
        sm.tk=t;

        measQmtx.lock();
        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl=measQ.front();
        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl_nr=measnoroadQ.front();
        measQ.pop();
        measnoroadQ.pop();
        measQmtx.unlock();

        pcl2eigen(Xpcl,sm.X1v);
        pcl2eigen(Xpcl_nr,sm.X1v_roadrem);



        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl_t (new pcl::PointCloud<pcl::PointXYZ> ());
        pcl::PointCloud<pcl::PointXYZ>::Ptr Xpcl_nr_t (new pcl::PointCloud<pcl::PointXYZ> ());
        if(H.isApprox(Eigen::Matrix4f::Identity())) {
                Xpcl_t=Xpcl;
                Xpcl_nr_t=Xpcl_nr;
                sm.X1gv=sm.X1v;
                sm.X1gv_roadrem=sm.X1v_roadrem;
        }
        else{
                pcl::transformPointCloud (*Xpcl, *Xpcl_t, H);
                pcl::transformPointCloud (*Xpcl_nr, *Xpcl_nr_t, H);

                pcl2eigen(Xpcl_t,sm.X1gv);
                pcl2eigen(Xpcl_nr_t,sm.X1gv_roadrem);


        }
        meas.push_back(Xpcl);
        meas_noroad.push_back(Xpcl_nr);







        return sm;
        // dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem

}

void MapLocalizer::addMeas(const Eigen::Ref<const Eigen::MatrixXf> &X,const Eigen::Ref<const Eigen::MatrixXf> &Xnoroad,float t){
        Timer TT("addMeas",timerptr);

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


        T.push_back(std::round(10e4*t)/10e4);

}


void MapLocalizer::addMap(const Eigen::Ref<const Eigen::MatrixXf> &X){
        Timer TT("addMap",timerptr);

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
        Timer TT("addMap2D",timerptr);

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

void MapLocalizer::computeHlevels(){
        Timer TT("computeHlevels",timerptr);

        Eigen::MatrixXf X1;
        pcl2eigen(map2D,X1);
        MatrixX2f X2 =X1(Eigen::all,Eigen::seq(0,1));
        bm.computeHlevels(X2);
}

int MapLocalizer::time2index(float tk){
        float tk1 = std::round(10e4*tk)/10e4;
        // float tk2 = std::round(10e5*tk)/10e5;
        // float tk3 = std::round(10e7*tk)/10e7;
        auto it1 = std::find(T.begin(), T.end(), tk1);
        // auto it2 = std::find(T.begin(), T.end(), tk2);
        // auto it3 = std::find(T.begin(), T.end(), tk3);
        if(it1!= T.end())
                return it1 - T.begin();
        // else if(it2!= T.end())
        //         return it2 - T.begin();
        // else if(it3!= T.end())
        //         return it3 - T.begin();
        else
                return -1;
}

void MapLocalizer::setgHk(float tk, Eigen::Matrix4f gHs ){
        int idx = time2index(tk);
        if (idx>=0) {
                gHkmtx.lock();
                gHk[idx] = gHs;
                gHkmtx.unlock();
        }
}
void MapLocalizer::setLookUpDist(std::string filename){
        Timer TT("setLookUpDist",timerptr);

        boost::filesystem::path p1(filename);


        float min_x,min_y,min_z,max_x,max_y,max_z;
        // octree->getBoundingBox(min_x,min_y,min_z,max_x,max_y,max_z);
        auto mnmx = MapPcllimits();
        min_x=mnmx(0)-100; min_y=mnmx(1)-100; min_z=mnmx(2)-50; max_x=mnmx(3)+100; max_y=mnmx(4)+100; max_z=mnmx(5)+50;

        Xdist_min.push_back(min_x);
        Xdist_min.push_back(min_y);
        Xdist_min.push_back(min_z);

        Xdist_max.push_back(max_x);
        Xdist_max.push_back(max_y);
        Xdist_max.push_back(max_z);

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
void MapLocalizer::setRegisteredSeqH_async(){
        seqregister_future = std::async(std::launch::async, &MapLocalizer::setRegisteredSeqH,this);
        // std::thread setRegisteredSeqH_thread(&MapLocalizer::setRegisteredSeqH,this);
        // setRegisteredSeqH_thread.detach();

}
void MapLocalizer::setRegisteredSeqH(){
        Timer TT("setRegisteredSeqH",timerptr);

        int n =meas.size();
        i1Himtx.lock();
        for(std::size_t i=0; i<n-1; ++i) {
                if(i1Hi_seq.find(i)==i1Hi_seq.end()) {

                        gicpseq.setInputSource(meas[i+1]);
                        gicpseq.setInputTarget(meas[i]);
                        pcl::PointCloud<pcl::PointXYZ>::Ptr resultXgicp(new pcl::PointCloud<pcl::PointXYZ>());
                        gicpseq.align(*resultXgicp);
                        auto H_gicp = gicpseq.getFinalTransformation();
                        i1Hi_seq[i][i+1]=H_gicp;

                }
        }
        i1Himtx.unlock();
        //std::cout << " done registering serq meas "<<std::endl;
        setSeq_gHk();

}
std::vector<Eigen::Matrix4f> MapLocalizer::setSeq_gHk(){
        Timer TT("setSeq_gHk",timerptr);

        // gHk.clear();
        // gHk.push_back(Eigen::Matrix4f::Identity());
        gHkmtx.lock();
        int n1 = gHk.size();
        for(std::size_t i=n1; i<meas.size(); ++i) {
                auto itp=i1Hi_seq.find(i-1);
                if(itp!=i1Hi_seq.end())
                        gHk.push_back(gHk[i-1]*itp->second[i]);
                else
                        break;
        }
        gHkmtx.unlock();
        setRelStates();
        //std::cout << " done gloabl gHk "<<std::endl;
        return gHk;
}

void MapLocalizer::setRelStates_async(){
        setrelstates_future = std::async(std::launch::async, &MapLocalizer::setRelStates,this);

}

void MapLocalizer::setRelStates(){
        Timer TT("setRelStates",timerptr);

        gHk = getSeq_gHk();

        // Vel.clear();
        // AngVel.clear();
        // XseqPos.clear();
        //
        // Vel.push_back(Eigen::Vector3f::Zero());
        // AngVel.push_back(Eigen::Vector3f::Zero());
        // XseqPos.push_back(Eigen::Vector3f::Zero());
        relStatemtx.lock();
        int n=XseqPos.size();
        for(std::size_t i=n; i<gHk.size(); ++i) {
                // if(gHk.find(i)==gHk.end())
                //         break;
                Eigen::Vector3f pm1 = gHk[i-1].col(3).head(3);
                Eigen::Vector3f p = gHk[i].col(3).head(3);
                XseqPos.push_back(p);

                auto dt = T[i]-T[i-1];

                Eigen::Vector3f v = p-pm1;
                Vel.push_back(v/dt);
                Eigen::Matrix4f H = gHk[i]*gHk[i-1].inverse();
                auto xrel = Hmat2pose_v2(H);
                // Eigen::Matrix3f R = H.block(0,0,3,3);
                // Eigen::Vector3f ea = R.eulerAngles(2, 1, 0);
                AngVel.push_back(xrel(Eigen::seq(3,5))/dt);



        }
        relStatemtx.unlock();

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
        Timer TT("getmaplocal",timerptr);

        pcl::CropBox<pcl::PointXYZ> boxFilter;
        boxFilter.setMin(Eigen::Vector4f(lb(0), lb(1), lb(2), 1.0));
        boxFilter.setMax(Eigen::Vector4f(ub(0), ub(1), ub(2), 1.0));
        boxFilter.setInputCloud(map);
        pcl::PointCloud<pcl::PointXYZ>::Ptr bodyFiltered(new  pcl::PointCloud<pcl::PointXYZ>());
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

Eigen::MatrixXf MapLocalizer::getmap2D_noroad_res_eigen(std :: vector<float> res,int dim){
        Timer TT("getmap2D_noroad_res_eigen",timerptr);


        pcl::PointCloud<pcl::PointXYZ>::Ptr mapcopy (new pcl::PointCloud<pcl::PointXYZ> ());
        pcl::copyPointCloud(*map2D,*mapcopy);
        if(dim==2) {
                for(auto &p:*mapcopy) {
                        p.z=0;
                }
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr outputcld (new pcl::PointCloud<pcl::PointXYZ> ());
        pcl_filter_cloud(mapcopy,outputcld,res);


        Eigen::MatrixXf X;
        pcl2eigen(outputcld,X);

        return X(Eigen::all,Eigen::seq(0,1));
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr MapLocalizer::getmap2D(){
        return map2D;
}
std::vector<Eigen::Vector3f> MapLocalizer::getvelocities(){
        relStatemtx.lock();
        std::vector<Eigen::Vector3f> V = Vel;
        relStatemtx.unlock();
        return V;
}

std::vector<Eigen::Vector3f> MapLocalizer::getpositions(){
        relStatemtx.lock();
        std::vector<Eigen::Vector3f> P = XseqPos;
        relStatemtx.unlock();
        return P;
}
std::vector<Eigen::Vector3f> MapLocalizer::getangularvelocities(){
        relStatemtx.lock();
        std::vector<Eigen::Vector3f> A = AngVel;
        relStatemtx.unlock();
        return A;
}



Eigen::VectorXf MapLocalizer::getLikelihoods_octree(const Eigen::Ref<const Eigen::MatrixXf> &Xposes,float tk){
        Timer TT("getLikelihoods_octree",timerptr);
        int idx = time2index(tk);
        // std::cout << "got Xposes size = " << Xposes.rows() << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr measfiltered;
        measfiltered=meas.at(idx);
        if(options["MeasMenaager"]["meas_Likelihood"]["downsample"]["enable"]==true) {
                std::vector<float> res = options["MeasMenaager"]["meas_Likelihood"]["downsample"]["resolution"];
                measfiltered.reset(new pcl::PointCloud<pcl::PointXYZ>());
                pcl_filter_cloud(meas.at(idx),measfiltered,res);
        }
        return computeLikelihood(octree,Xposes,measfiltered,options["Localize"]["dmax"],options["Localize"]["sig0"]);
}

Eigen::VectorXf MapLocalizer::getLikelihoods_lookup(const Eigen::Ref<const Eigen::MatrixXf> &Xposes,float tk){
        Timer TT("getLikelihoods_lookup",timerptr);
        int idx = time2index(tk);

        pcl::PointCloud<pcl::PointXYZ>::Ptr measfiltered;
        measfiltered=meas.at(idx);
        if(options["MeasMenaager"]["meas_Likelihood"]["downsample"]["enable"]==true) {
                std::vector<float> res = options["MeasMenaager"]["meas_Likelihood"]["downsample"]["resolution"];
                measfiltered.reset(new pcl::PointCloud<pcl::PointXYZ>());
                pcl_filter_cloud(meas.at(idx),measfiltered,res);
        }
        return computeLikelihood_lookup(Xdist,options["Localize"]["lookup"]["resolution"],Xdist_min,Xdist_max,
                                        Xposes,measfiltered,options["Localize"]["dmax"],options["Localize"]["sig0"]);

}


std::vector<Eigen::Matrix4f> MapLocalizer::getSeq_gHk(){
        gHkmtx.lock();
        std::vector<Eigen::Matrix4f> G=gHk;
        gHkmtx.unlock();
        return G;

}

std::vector<Eigen::Matrix4f> MapLocalizer::geti1Hi_seq_vec(){
        Timer TT("geti1Hi_seq_vec",timerptr);

        std::vector<Eigen::Matrix4f> Ht;
        i1Himtx.lock();
        for(std::size_t i=0; i<meas.size()-1; ++i) {
                if(i1Hi_seq.find(i)!=i1Hi_seq.end())
                        Ht.push_back( i1Hi_seq[i][i+1] );
        }
        i1Himtx.unlock();

        return Ht;
}
std::unordered_map<int, std::unordered_map<int,Eigen::Matrix4f> >  MapLocalizer::geti1Hi_seq(){
        i1Himtx.lock();
        std::unordered_map<int, std::unordered_map<int,Eigen::Matrix4f> > Ht = i1Hi_seq;
        i1Himtx.unlock();

        return Ht;
}

std::vector<Eigen::Matrix4f> MapLocalizer::getsetSeq_gHk(float tk, Eigen::Matrix4f gHkatk){
        Timer TT("getsetSeq_gHk",timerptr);
        int tidx = time2index(tk);

        setRegisteredSeqH();
        auto gHkset = getSeq_gHk();
        auto i1Hi_seqset = geti1Hi_seq();
        // for (auto const &pair: i1Hi_seq) {
        //         for (auto const &pair2: pair.second) {
        //                 std::cout << "{" << pair.first << "--> " << pair2.first << "}\n";
        //         }
        // }

        // std::cout<< "gHkset 1 size = "<< gHkset.size()<<std::endl;
        if(tidx>=gHkset.size() || tidx<0)
                return gHkset;

        gHkset.at(tidx) = gHkatk;
        // std::cout << " at(tk) done" << std::endl;
        if(tidx<gHkset.size()-1) {
                for(std::size_t k=tidx+1; k<gHkset.size(); ++k) {
                        gHkset[k]=gHkset[k-1]*i1Hi_seqset[k-1][k];
                        // std::cout<< ">>gHkset k = "<< k << " / " << meas.size() <<std::endl;
                }
        }
        // std::cout<< "gHkset 2 size = "<< gHkset.size()<<std::endl;
        if(tidx>0) {
                for(int k=tidx-1; k>=0; --k) {
                        auto H = i1Hi_seqset[k][k+1].inverse();
                        gHkset[k]=gHkset[k+1]*H;
                        // std::cout<< "gHkset k = "<< k << " / " << meas.size() <<std::endl;
                }
        }
        //std::cout<< "gHkset 3 size = "<< gHkset.size()<<std::endl;
        return gHkset;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr MapLocalizer::getalignSeqMeas(float t0,float tf,float tk, Eigen::Matrix4f gHkatk,std::vector<float> res,int dim){
        Timer TT("getalignSeqMeas",timerptr);
        int tkidx = time2index(tk);
        int t0idx = time2index(t0);
        int tfidx = time2index(tf);

        auto gHkset=getsetSeq_gHk(tk, gHkatk);
        pcl::PointCloud<pcl::PointXYZ>::Ptr mergedpcl(new pcl::PointCloud<pcl::PointXYZ>());
        for(std::size_t k=t0idx; k<=tfidx; ++k) {
                // std::cout<< "k="<<k<<std::endl;
                auto pclptr = getmeas(k);

                pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
                pcl::transformPointCloud (*pclptr, *transformed_cloud, gHkset[k]);

                *mergedpcl=*mergedpcl+*transformed_cloud;
        }
        if(dim==2) {
                for(auto &p:*mergedpcl) {
                        p.z=0;
                }
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr outputcld (new pcl::PointCloud<pcl::PointXYZ> ());
        pcl_filter_cloud(mergedpcl,outputcld,res);
        return outputcld;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr MapLocalizer::getalignSeqMeas_noroad(float t0,float tf,float tk, Eigen::Matrix4f gHkatk,std::vector<float> res,int dim){
        Timer TT("getalignSeqMeas_noroad",timerptr);
        int tkidx = time2index(tk);
        int t0idx = time2index(t0);
        int tfidx = time2index(tf);

        // std::cout << "getalignSeqMeas_noroad----------" << std::endl;
        // std::cout << "t0,tf,tk = "<< t0 << " "<< tf << " "<< tk << " " << std::endl;
        auto gHkset=getsetSeq_gHk(tk, gHkatk);
        // std::cout << "getsetSeq_gHk----DONE------" << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr mergedpcl(new pcl::PointCloud<pcl::PointXYZ>());
        for(std::size_t k=t0idx; k<=tfidx; ++k) {
                // std::cout<< "k="<<k<<std::endl;
                auto pclptr = getmeas_noroad(k);

                pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
                pcl::transformPointCloud (*pclptr, *transformed_cloud, gHkset[k]);

                *mergedpcl=*mergedpcl+*transformed_cloud;
        }
        if(dim==2) {
                for(auto &p:*mergedpcl) {
                        p.z=0;
                }
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr outputcld (new pcl::PointCloud<pcl::PointXYZ> ());;
        pcl_filter_cloud(mergedpcl,outputcld,res);
        return outputcld;
}

Eigen :: MatrixXf MapLocalizer::getalignSeqMeas_eigen(float t0,float tf,float tk, Eigen::Matrix4f gHkest,std::vector<float> res,int dim){
        Timer TT("getalignSeqMeas_eigen",timerptr);
        int tkidx = time2index(tk);
        int t0idx = time2index(t0);
        int tfidx = time2index(tf);

        //std::cout << res[0] << res[1] << res[2]<<std::endl;
        auto outputcld = getalignSeqMeas(t0,tf,tk,gHkest,res,dim);
        //std::cout << "outputcld = "<< outputcld->size() <<std::endl;

        Eigen :: MatrixXf X=Eigen :: MatrixXf::Zero(outputcld->size(),3);
        pcl2eigen(outputcld,X);
        return X(Eigen::all,Eigen::seq(0,dim-1));
}
Eigen :: MatrixXf MapLocalizer::getalignSeqMeas_noroad_eigen(float t0,float tf,float tk, Eigen::Matrix4f gHkest,std::vector<float> res,int dim){
        Timer TT("getalignSeqMeas_noroad_eigen",timerptr);
        int tkidx = time2index(tk);
        int t0idx = time2index(t0);
        int tfidx = time2index(tf);

        //std::cout << res[0] << res[1] << res[2]<<std::endl;
        auto outputcld = getalignSeqMeas_noroad(t0,tf,tk,gHkest,res,dim);
        //std::cout << "outputcld = "<< outputcld->size() <<std::endl;

        Eigen :: MatrixXf X=Eigen :: MatrixXf::Zero(outputcld->size(),3);
        pcl2eigen(outputcld,X);
        return X(Eigen::all,Eigen::seq(0,dim-1));

}

//-----------------Aligners-------------------------
void
MapLocalizer::BMatchseq_async(float t0,float tf,float tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHkest,bool gicp){
// BMatchAndCorrH

        if (bmHsols_async_future.valid()==false) {
                bmHsols_async_future = std::async(std::launch::async, &MapLocalizer::BMatchseq_async_caller,this,t0,tf,tk,gHkest,gicp);
        }
        else{
                std::cout << "bmHsols_async_future valid is true...something could be running" << std::endl;
        }
}
BMatchAndCorrH_async
MapLocalizer::getBMatchseq_async(){
        if (bmHsols_async_future.valid()==false) {
                std::cout << "bmHsols_async_future valid is false...NOTHING could be running" << std::endl;
                BMatchAndCorrH_async bmHsols_async_result;
                bmHsols_async_result.isDone=false;
                return bmHsols_async_result;
        }
        std::future_status status = bmHsols_async_future.wait_for(10ms);
        if( status==std::future_status::timeout ) {
                std::cout << "bmHsols_async_future timedout" << std::endl;
                BMatchAndCorrH_async bmHsols_async_result;
                bmHsols_async_result.isDone=false;
                return bmHsols_async_result;
        }
        if( status==std::future_status::ready ) {
                BMatchAndCorrH_async bmHsols_async_result= bmHsols_async_future.get();
                return bmHsols_async_result;
        }
}


BMatchAndCorrH_async
MapLocalizer::BMatchseq_async_caller(float t0,float tf,float tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHkest,bool gicp){

        Timer TT("BMatchseq_async_caller",timerptr);


        BMatchAndCorrH_async B;

        B.tk=tk;
        B.t0=t0;
        B.tf=tf;
        B.do_gicp=gicp;
        B.gHkest_initial=gHkest;
        std::chrono::time_point<std::chrono::system_clock> st = std::chrono::system_clock::now();
        B.bmHsol = BMatchseq(t0,tf,tk,gHkest,gicp);
        std::chrono::time_point<std::chrono::system_clock> et = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = et-st;
        B.time_taken = elapsed_seconds.count();
        B.isDone=true;
        B.bmHsol.isDone=true;

        return B;

}

BMatchAndCorrH
MapLocalizer::BMatchseq(float t0,float tf,float tk,const Eigen::Ref<const Eigen :: Matrix4f>&gHkest,bool gicp){
        Timer TT("BMatchseq",timerptr);

        std::vector<float> res({bm.dxMatch(0),bm.dxMatch(1),1});
        // Vector6f xpose=Hmat2pose(gHkest);
        Vector6f xpose=Hmat2pose_v2(gHkest);
        Eigen :: Matrix4f HI = Eigen :: Matrix4f::Identity();
        pcl::PointCloud<pcl::PointXYZ>::Ptr Xsrcpcl= getalignSeqMeas_noroad(t0,tf,tk, HI,res,2);
        //std::cout << "Xsrcpcl.size = " << Xsrcpcl->size() << std::endl;

        Eigen::MatrixXf Xsrc;
        pcl2eigen(Xsrcpcl,Xsrc);
        //std::cout << "XsrcXsrc.rows = " << Xsrc.rows() << std::endl;
        MatrixX2f Xsrc2D= Xsrc(Eigen::all,Eigen::seq(0,1));
        //std::cout << "Xsrc2D.rows = " << Xsrc2D.rows() << std::endl;




        Eigen::Matrix3f Rz;
        Rz = Eigen::AngleAxisf(xpose(3), Eigen::Vector3f::UnitZ());
        //std::cout << "Rz = " << Rz <<std::endl;


        Eigen::Matrix3f H12=Eigen::Matrix3f::Identity();
        H12.block(0,0,2,2) = Rz.block(0,0,2,2);
        H12(0,2) = xpose(0);
        H12(1,2) = xpose(1);
        //std::cout << "H12 = " << H12 <<std::endl;

        BMatchAndCorrH solret;
        solret.sols =bm.getmatch(Xsrc2D,H12);
        //std::cout << "Dones sols of bm match = " <<std::endl;
        //std::cout << "solret.sols[0].H = " << solret.sols[0].H << std::endl;
        for(int si=0; si<solret.sols.size(); ++si) {
                Eigen::Matrix4f Hbm=Eigen::Matrix4f::Identity();
                Hbm.block(0,0,2,2)=solret.sols[si].H.block(0,0,2,2);
                Hbm(0,3)=solret.sols[si].H(0,2);
                Hbm(1,3)=solret.sols[si].H(1,2);
                Vector6f xposeBM=Hmat2pose_v2(Hbm);
                //std::cout << "xposeBM = " << xposeBM << std::endl;
                // Eigen::Matrix3f R =Eigen::Matrix3f::Identity();
                // R.block(0,0,2,2)=solret.sols[0].H.block(0,0,2,2);
                // Eigen::Vector3f vv2;
                // vv2 = R.eulerAngles(2, 1, 0);
                xpose(3)=xposeBM(3);
                xpose(0)=xposeBM(0);
                xpose(1)=xposeBM(1);
                //std::cout << "xpose = " << xpose << std::endl;
                auto gHkcorr=pose2Hmat(xpose);

                if(gicp==true) {
                        std::vector<float> res = options["mapfit"]["downsample"]["resolution"];
                        Eigen :: Matrix4f HI = Eigen :: Matrix4f::Identity();
                        pcl::PointCloud<pcl::PointXYZ>::Ptr Xsrcpcl= getalignSeqMeas(t0,tf,tk, HI,res,3);
                        gHkcorr = gicp_correction(Xsrcpcl, gHkcorr);
                }

                solret.gHkcorr.push_back(gHkcorr);
        }
        //std::cout << "solret.gHkcorr = " << solret.gHkcorr << std::endl;


        return solret;

}

Vector6f
MapLocalizer::gicp_correction_pose(float tk,Vector6f xpose){

        auto gHkcorr=pose2Hmat(xpose);

        std::vector<float> res = options["mapfit"]["downsample"]["resolution"];
        Eigen :: Matrix4f HI = Eigen :: Matrix4f::Identity();
        pcl::PointCloud<pcl::PointXYZ>::Ptr Xsrcpcl= getalignSeqMeas(tk,tk,tk, HI,res,3);
        gHkcorr = gicp_correction(Xsrcpcl, gHkcorr);
        Vector6f xposecorr=Hmat2pose_v2(gHkcorr);

        return xposecorr;
}

// gHk takes k-frame local to gloal inertial frame
Eigen::Matrix4f
MapLocalizer::gicp_correction(pcl::PointCloud<pcl::PointXYZ>::Ptr Xsrcpcl, const Eigen::Ref<const Eigen :: Matrix4f>&gHk_est){
        Timer TT("gicp_correction",timerptr);


        pcl::PointCloud<pcl::PointXYZ>::Ptr resultXgicp(new pcl::PointCloud<pcl::PointXYZ>(Xsrcpcl->size(),1));
        pcl::transformPointCloud (*Xsrcpcl, *resultXgicp, gHk_est);

        pcl::PointXYZ min_pt,max_pt;
        pcl::getMinMax3D (*Xsrcpcl, min_pt, max_pt);
        //std::cout << "gicp min pt = "<< min_pt.x << " "<< min_pt.y << " "<< min_pt.z << " "<<std::endl;
        float extra=50;
        min_pt.x-=extra; min_pt.y-=extra; min_pt.z-=extra;
        max_pt.x+=extra; max_pt.y+=extra; max_pt.z+=extra;
        //std::cout << "gicp min pt extra = "<< min_pt.x << " "<< min_pt.y << " "<< min_pt.z << " "<<std::endl;
        //std::cout << "gicp max pt extra = "<< max_pt.x << " "<< max_pt.y << " "<< max_pt.z << " "<<std::endl;
        auto maplocal = getmaplocal(min_pt,max_pt);

        std::vector<float> res = options["mapfit"]["downsample"]["resolution"];
        pcl::PointCloud<pcl::PointXYZ>::Ptr maplocal_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl_filter_cloud(maplocal,maplocal_filtered,res);


        //std::cout << "maplocal_filtered.size = "<< maplocal_filtered->size()<<std::endl;

        gicp.setInputTarget(maplocal_filtered);
        gicp.setInputSource(resultXgicp);

        pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>());
        gicp.align(*result);
        auto H_gicp = gicp.getFinalTransformation();
        //std::cout << "H_gicp = " << H_gicp << std::endl;
        return H_gicp*gHk_est;
}


timerdict
MapLocalizer::gettimers(){
        return *timerptr;
}
