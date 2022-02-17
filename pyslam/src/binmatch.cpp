#include "binmatch.h"

void takejson(const nlohmann::json& json){
        std::cout << "I got json as :"<<std::endl;
        std::cout << json << std::endl;


}
MatrixXXi
UpsampleMax(const Eigen::Ref<const MatrixXXi>& Hup,int n){
        // printmsg("Hup.rows()",Hup.rows());
        // printmsg("Hup.cols()",Hup.cols());
        MatrixXXi H=MatrixXXi::Zero(Hup.rows()/2,Hup.cols()/2);
        // printmsg("H.rows()",H.rows());
        // printmsg("H.cols()",H.cols());
        for(int j=0; j<H.rows(); ++j) {
                for(int k=0; k<H.cols(); ++k) {
                        int lbx=std::max(2*j,0);
                        int ubx=std::min(2*j+n+1,int(Hup.rows()));
                        int lby=std::max(2*k,0);
                        int uby=std::min(2*k+n+1,int(Hup.cols()));
                        // printmsg("lbx",lbx);
                        // printmsg("ubx",ubx);
                        // printmsg("lby",lby);
                        // printmsg("uby",uby);
                        // H(j,k) = Hup(Eigen::seq(lbx,ubx),Eigen::seq(lby,uby)).maxCoeff();
                        H(j,k) = Hup.block(lbx,lby,ubx-lbx,uby-lby).maxCoeff();
                }

        }
        return H;
}


MatrixXXi
computeHitogram2D(const Eigen::Ref<const MatrixX2f>& X,Matrix2irow n_edges, Matrix2frow xymin, Matrix2frow xymax){
        // Lxy is the length

        MatrixXXi H = MatrixXXi::Zero(n_edges(0)-1,n_edges(1)-1);

        Matrix2frow dxy({xymax(0)-xymin(0),xymax(1)-xymin(1)});
        // Matrix2frow xymin({xmin(0),xmin(1)});
        // Matrix2frow xymax({xmax(0),xmax(1)});

        Matrix2frow nxy({n_edges(0)-1,n_edges(1)-1});
        // MatrixX2f X1 = X;
        MatrixX2f X1 = X.rowwise()-xymin;
        MatrixX2f X2 = -(X.rowwise()-xymax);


        MatrixXbcol inbnd= ( (X1.array()>=0 ).array() * (X2.array()>=0 ).array()).rowwise().all();

        MatrixX2i dd =( ( (X1.array().rowwise())/dxy.array() ).array().rowwise()*nxy.array() ).cast<int>();
        // dd = dd.array().rowwise()*nxy.array();

        for(int i=0; i< X.rows(); ++i) {
                if (inbnd(i))
                        H(dd(i,0),dd(i,1))+=1;
        }

        return H;
}



int
getPointCost(const Eigen::Ref<const MatrixXXi>& H, const Eigen::Ref<const Matrix2frow>& dx,
             const Eigen::Ref<const Eigen::MatrixX2f>& X,const Eigen::Ref<const Matrix2frow>& Oj){

        MatrixX2i Pn= ( (X.rowwise()+Oj).array().rowwise()/dx.array() ).cast<int>();

        Matrix2irow xymin({0,0});
        Matrix2irow xymax({H.rows(),H.cols()});
        MatrixXbcol inbnd= ( ((Pn.rowwise()-xymin).array()>=0 ).array() * ((Pn.rowwise()-xymax).array()<0 ).array() ).rowwise().all();

        int c=0;
        for (int k=0; k<Pn.rows(); ++k) {
                if(inbnd(k)>0)
                        c+=H(Pn(k,0),Pn(k,1));
        }
        return c;
}



BBox
SolBox2BBox(const SolBox& solbox){
        BBox bbox;
        bbox.x1 = solbox.lb(0);
        bbox.y1 = solbox.lb(1);

        auto ub = solbox.lb+solbox.dx;
        bbox.x2 = ub(0);
        bbox.y2 = ub(1);

        return bbox;
}

std::vector<SolBox>
quadSplitSolBox(const SolBox& solbox){
        std::vector<SolBox> v;
        SolBox b1,b2,b3,b4;
        b1 = solbox;
        b2 = solbox;
        b3 = solbox;
        b4 = solbox;

        b1.lb = solbox.lb;
        b1.dx=solbox.dx/2;
        b1.flg=false;
        v.push_back(b1);

        b2.lb = solbox.lb; //+(Matrix2frow({1,0}).array()*solbox.dx.array())/2;
        b2.lb(0)=b2.lb(0)+solbox.dx(0)/2;
        b2.dx=solbox.dx/2;
        b2.flg=false;
        v.push_back(b2);

        b3.lb = solbox.lb; //+(Matrix2frow({1,0}).array()*solbox.dx.array())/2;
        b3.lb(1)=b3.lb(1)+solbox.dx(1)/2;
        b3.dx=solbox.dx/2;
        b3.flg=false;
        v.push_back(b3);

        b4.lb = solbox.lb+solbox.dx/2;
        b4.dx=solbox.dx/2;
        b4.flg=false;
        v.push_back(b4);

        return v;
}

bool
SolBoxesIntersect(const BBox& bb1,const SolBox& sb2){

        auto bb2=SolBox2BBox(sb2);

        auto x_left = std::max(bb1.x1, bb2.x1);
        auto y_bottom = std::max(bb1.y1, bb2.y1);
        auto x_right = std::min(bb1.x2, bb2.x2);
        auto y_top = std::min(bb1.y2, bb2.y2);

        if ( (x_right < x_left) || (y_top < y_bottom))
                return false;
        else
                return true;
}

bool
SolBoxesIntersect(const SolBox& sb1,const SolBox& sb2){
        auto bb1=SolBox2BBox(sb1);
        return SolBoxesIntersect(bb1,sb2);
}

std::ostream& operator<<(std::ostream& os, const SolBox& sb)
{
        os << "lb = " << sb.lb <<  ", dx = " << sb.dx << " ,cost = " << sb.cost << ", lvl = " << sb.lvl << ", th = "<< sb.th << std::endl;
        return os;
}

std::ostream& operator<<(std::ostream& os, const BBox& sb)
{
        os << "(x1,y1) = (" << sb.x1 <<  "," << sb.y1 << ") , (x2,y2) = (" << sb.x1 <<  "," << sb.y1 <<")" << std::endl;
        return os;
}

std::ostream& operator << (std::ostream& os, const std::vector<SolBox>& v)
{
        os << "[";
        for (auto sb : v)
        {
                os << sb << std::endl;
        }
        os << "]";
        return os;
}


BinMatch::BinMatch(const std::string & opt){
        setOptions(opt);
}
void BinMatch::setOptions(const std::string &opt){
        options=json::parse(opt);
        Lmax=Matrix2frow(options["BinMatch"]["Lmax"][0],options["BinMatch"]["Lmax"][1]);
        dxMatch=Matrix2frow(options["BinMatch"]["dxMatch"][0],options["BinMatch"]["dxMatch"][1]);
        dxBase=Matrix2frow(options["BinMatch"]["dxBase"][0],options["BinMatch"]["dxBase"][1]);
        thmax=options["BinMatch"]["thmax"];
        thfineres=options["BinMatch"]["thfineres"];
}

void BinMatch::computeHlevels(const Eigen::Ref<const MatrixX2f>& Xtarg){

        Matrix2frow mn=Matrix2frow::Zero();
        Matrix2frow mx=Matrix2frow::Zero();
        mn_orig=Matrix2frow::Zero();
        // Eigen::Matrix3f H21comp=Eigen::Matrix3f::Identity();


        mn_orig(0) = Xtarg.col(0).minCoeff();
        mn_orig(1) = Xtarg.col(1).minCoeff();
        mn_orig=mn_orig-dxMatch;


        MatrixX2f Xtarg1=Xtarg.rowwise()-mn_orig;

        mn(0) = Xtarg1.col(0).minCoeff();
        mn(1) = Xtarg1.col(1).minCoeff();
        mx(0) = Xtarg1.col(0).maxCoeff();
        mx(1) = Xtarg1.col(1).maxCoeff();



        // Matrix2frow P = mx-mn;

        int mxlvl=0;
        // Matrix2frow dx0=mx+dxMatch;


        int f;
        for(std::size_t i=0; i<100; ++i) {
                f=std::pow(2,i);
                dxlevels.emplace_back(Matrix2frow ({ (mx(0)+1*dxMatch(0))/f, (mx(1)+1*dxMatch(1))/f}));

                if ( (dxlevels.back().array()<=dxMatch.array()).array().any() )
                        break;
        }

        mxlvl=dxlevels.size();
        Matrix2irow n_edges = ( (mx+1*dxMatch).array()/dxlevels.back().array()+1 ).cast<int>();
        MatrixXXi H1match= computeHitogram2D(Xtarg1,n_edges, Matrix2frow({0,0}), mx+1*dxMatch);

        // printmsg("computeHitogram2D","computeHitogram2D is done");
        // printmsg("H1match rowwise max",H1match.rowwise().maxCoeff() );

        MatrixXXi H2match = H1match.unaryExpr(
                [](int x) {
                return ((x>0) ? 1 : 0 );
        });

        // printmsg("H2match","H2match is done");

        HLevels.push_back(H2match);
        int n=2;

        // printmsg("mxlvl = ",mxlvl);

        for(int i=1; i<mxlvl; ++i) {
                // printmsg("UpsampleMax i",i);
                auto Hup = HLevels.back();
                auto H=UpsampleMax(Hup,n);
                HLevels.push_back(H);
        }

        // printmsg("HLevels[0]",HLevels[0]);

        mxLVL=int(HLevels.size())-1;
        std::reverse(HLevels.begin(),HLevels.end());



}

std::vector<BinMatchSol>
BinMatch::getmatch(const Eigen::Ref<const MatrixX2f>& Xsrc,const Eigen::Ref<const Eigen::Matrix3f>& H12){
        H12mn = H12;
        // H12mn.block(0,2,2,0)=H12mn.block(0,2,2,0)-mn_orig.matrix().transpose();
        H12mn(0,2)=H12mn(0,2)-mn_orig(0);
        H12mn(1,2)=H12mn(1,2)-mn_orig(1);

        t0(0)= H12mn(0,2);
        t0(1)= H12mn(1,2);
        // Matrix2frow L0=t0-Lmax;
        // Matrix2frow L1=t0+Lmax;



        SolBox solbox_init;
        solbox_init.lb = Matrix2frow({0,0});
        solbox_init.dx = dxlevels[0];
        solbox_init.cost=0;
        solbox_init.lvl=0;
        solbox_init.th=0;
        solbox_init.flg=false; //


        // int lvl=0;
        // auto dx=dxlevels[lvl];
        // auto H=HLevels[lvl];

        std::vector<SolBox> qv;
        for(float th=-thmax; th<thmax; th=th+thfineres) {

                Eigen::Matrix2f R ({{std::cos(th), -std::sin(th)},{std::sin(th), std::cos(th)}});
                MatrixX2f XX1 = (R*(Xsrc.transpose())).transpose();
                MatrixX2f XX=((H12mn.block(0,0,2,2))*(XX1.transpose())).transpose();
                Xth[th]=XX;

                SolBox sb = solbox_init;
                sb.th = th;
                qv.push_back(sb);
                // std::cout <<"sb = " << sb<<std::endl;
                // auto qsv = quadSplitSolBox(sb);
                // std::cout << qsv<<std::endl;

        }



        BBox bb1;
        bb1.x1=t0(0)-Lmax(0);
        bb1.y1=t0(1)-Lmax(1);
        bb1.x2=t0(0)+Lmax(0);
        bb1.y2=t0(1)+Lmax(1);
        std::cout << "bb1.x1,bb1.y1,bb1.x2,bb1.y2 = " << bb1.x1 << " " << bb1.y1 << " " << bb1.x2 << " " << bb1.y2 << std::endl;

        auto cmp = [](SolBox left, SolBox right) {
                           return (left.cost) < (right.cost);
                   };

        // now break the initial box further according to dxBase
        int cnt=0;
        if (dxBase(0)>0) {
                while(1) {
                        cnt=0;
                        for(std::size_t i=0; i<qv.size(); ++i) {
                                if ((qv[i].dx.array()>dxBase.array()).array().any()) {
                                        auto qsv = quadSplitSolBox(qv[i]);
                                        for(std::size_t j=0; j<qsv.size(); ++j) {
                                                if(SolBoxesIntersect(bb1,qsv[j])) {
                                                        qsv[j].lvl = qv[i].lvl+1;
                                                        qv.push_back(qsv[j]);
                                                }
                                        }
                                        qv[i].cost=-1;
                                        cnt++;
                                }
                        }
                        if(cnt>0) {
                                std::sort(qv.begin(), qv.end(),cmp);
                                qv.erase(qv.begin()+0,qv.begin()+cnt);
                        }
                        else
                                break;
                }
        }



        #pragma omp parallel for num_threads(8)
        for(std::size_t i=0; i<qv.size(); ++i) {
                qv[i].cost=getPointCost(HLevels[qv[i].lvl],dxlevels[qv[i].lvl],Xth[qv[i].th],qv[i].lb);
                qv[i].flg=true;
        }

        qvinit=qv;
        std::make_heap(qvinit.begin(), qvinit.end(),cmp);

        std::priority_queue<SolBox, std::deque<SolBox>, decltype(cmp)> q(cmp);
        for(std::size_t i=0; i<qv.size(); ++i) {
                q.push(qv[i]);
        }
        qv.clear();

        MatrixX2f XX0=((H12mn.block(0,0,2,2))*(Xsrc.transpose())).transpose();
        XX0 = XX0.rowwise()+t0;
        int cost0 = getPointCost(HLevels[mxLVL],dxlevels[mxLVL],XX0,Matrix2frow({0,0}));

        std::vector<SolBox> qvMxLvL;
        qvMxLvL.reserve(200);
        qv.reserve(100);
        while(1) {

                qv.clear();
                qvMxLvL.clear();
                for(std::size_t i=0; i<200; ++i) {
                        if (q.empty())
                                break;
                        SolBox sb=q.top();
                        if(sb.lvl<mxLVL) {
                                auto qsv = quadSplitSolBox(sb);
                                for(std::size_t j=0; j<qsv.size(); ++j) {
                                        if(SolBoxesIntersect(bb1,qsv[j])) {
                                                qsv[j].lvl = sb.lvl+1;
                                                qv.push_back(qsv[j]);
                                        }
                                }
                        }
                        else{
                                qvMxLvL.push_back(sb);
                        }
                        q.pop();
                }
                #pragma omp parallel for num_threads(6)
                for(std::size_t i=0; i<qv.size(); ++i) {
                        qv[i].cost=getPointCost(HLevels[qv[i].lvl],dxlevels[qv[i].lvl],Xth[qv[i].th],qv[i].lb);
                        qv[i].flg=true;
                }
                for(std::size_t i=0; i<qv.size(); ++i) {
                        q.push(qv[i]);
                }
                for(std::size_t i=0; i<qvMxLvL.size(); ++i) {
                        q.push(qvMxLvL[i]);
                }

                SolBox sb=q.top();
                if(sb.lvl==mxLVL) {
                        break;
                }
                q.pop();
                auto qsv = quadSplitSolBox(sb);
                for(std::size_t j=0; j<qsv.size(); ++j) {
                        if(SolBoxesIntersect(bb1,qsv[j])) {
                                qsv[j].lvl = sb.lvl+1;
                                qsv[j].cost=getPointCost(HLevels[qsv[j].lvl],dxlevels[qsv[j].lvl],Xth[qsv[j].th],qsv[j].lb);
                                q.push(qsv[j]);
                        }
                }

        }
        auto bestcst = q.top().cost;
        std::vector<BinMatchSol> finalsols;
        finalsols.reserve(30);
        while(1) {
                if(q.top().cost==bestcst) {
                        SolBox sb=q.top();
                        q.pop();

                        auto t=sb.lb-t0;

                        Eigen::Matrix3f HcompR = Eigen::Matrix3f::Identity();
                        Eigen::Matrix2f R ({{std::cos(sb.th), -std::sin(sb.th)},{std::sin(sb.th), std::cos(sb.th)}});
                        HcompR.block(0,0,2,2)=R;

                        Eigen::Matrix3f Ht= Eigen::Matrix3f::Identity();
                        Ht(0,2)=t(0);
                        Ht(1,2)=t(1);

                        Eigen::Matrix3f H12comp = (Ht*H12mn)*HcompR;
                        H12comp(0,2)+=mn_orig(0);
                        H12comp(1,2)+=mn_orig(1);

                        // Eigen::Matrix3f H21comp=H12comp.inverse()

                        BinMatchSol bms;
                        bms.H=H12comp;
                        bms.cost0=cost0;
                        bms.cost=sb.cost;
                        bms.lvl = sb.lvl;
                        bms.mxLVL = mxLVL;
                        finalsols.push_back(bms);
                }
                else
                        break;


        }

        return finalsols;
}
