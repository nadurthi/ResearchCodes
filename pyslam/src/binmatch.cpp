#include "binmatch.h"


MatrixXXi
UpsampleMax(const Eigen::Ref<const MatrixXXi>& Hup,int n){
        MatrixXXi H=MatrixXXi::Zero(Hup.rows()/2,Hup.cols()/2);
        for(int j=0; j<H.rows(); ++j) {
                for(int k=0; j<H.cols(); ++k) {
                        int lbx=std::max(2*j,0);
                        int ubx=std::min(2*j+n,int(Hup.rows())-1)+1;
                        int lby=std::max(2*k,0);
                        int uby=std::min(2*k+n,int(Hup.cols())-1)+1;
                        H(j,k) = Hup(Eigen::seq(lbx,ubx),Eigen::seq(lby,uby)).maxCoeff();
                }

        }
        return H;
}


MatrixXXi
computeHitogram2D(const Eigen::Ref<const MatrixX2f>& X,Matrix2irow n_edges, Matrix2frow xmin, Matrix2frow xmax){
        // Lxy is the length

        MatrixXXi H = MatrixXXi::Zero(n_edges(0)-1,n_edges(1)-1);

        Matrix2frow dxy({xmax(0)-xmin(0),xmax(1)-xmin(1)});
        Matrix2frow xymin({xmin(0),xmin(1)});
        Matrix2frow xymax({xmax(0),xmax(1)});

        Matrix2irow nxy({n_edges(0)-1,n_edges(1)-1});
        // MatrixX2f X1 = X;
        MatrixX2f X1 = X.rowwise()-xymin;
        MatrixX2f X2 = -(X.rowwise()-xymax);


        MatrixXbcol inbnd= ( (X1.array()>=0 ).array() * (X2.array()>=0 ).array()).rowwise().all();

        MatrixX2i dd =( (X1.array().rowwise())/dxy.array() ).cast<int>();
        dd = dd.array().rowwise()*nxy.array();

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
quadSplitSolBox(SolBox solbox){
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

        b2.lb = solbox.lb; //+(Matrix2frow({1,0}).array()*solbox.dx.array())/2;
        b2.lb(1)=b2.lb(1)+solbox.dx(1)/2;
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
SolBoxesIntersect(BBox bb1,SolBox sb2){

        auto bb2=SolBox2BBox(sb2);

        auto x_left = std::max(bb1.x1, bb2.x1);
        auto y_top = std::max(bb1.y1, bb2.y1);
        auto x_right = std::min(bb1.x2, bb2.x2);
        auto y_bottom = std::min(bb1.y2, bb2.y2);

        if ( (x_right < x_left) || (y_bottom < y_top))
                return false;
        else
                return true;
}

bool
SolBoxesIntersect(SolBox sb1,SolBox sb2){
        auto bb1=SolBox2BBox(sb1);
        return SolBoxesIntersect(bb1,sb2);
}




BinMatch::BinMatch(const Eigen::Ref<const Matrix2frow>& Lmax_,
                   const Eigen::Ref<const Matrix2frow>& dxMatch_,
                   const Eigen::Ref<const Matrix2frow>& dxBase_,std::string opt){
        Lmax=Lmax_;
        dxMatch=dxMatch_;
        dxBase=dxBase_;
        options=json::parse(opt);
}

void BinMatch::computeHlevels(const Eigen::Ref<const MatrixX2f>& Xtarg){
        int n=2;
        Matrix2frow mn=Matrix2frow::Zero();
        Matrix2frow mx=Matrix2frow::Zero();
        mn_orig=Matrix2frow::Zero();
        Eigen::Matrix3f H21comp=Eigen::Matrix3f::Identity();


        mn_orig(0) = Xtarg.col(0).minCoeff();
        mn_orig(1) = Xtarg.col(1).minCoeff();
        mn_orig=mn_orig-dxMatch;


        MatrixX2f Xtarg1=Xtarg.rowwise()-mn_orig;

        mn(0) = Xtarg1.col(0).minCoeff();
        mn(1) = Xtarg1.col(1).minCoeff();
        mx(0) = Xtarg1.col(0).maxCoeff();
        mx(1) = Xtarg1.col(1).maxCoeff();



        Matrix2frow P = mx-mn;

        int mxlvl=0;
        Matrix2frow dx0=mx+dxMatch;


        int f;
        for(int i=0; i<100; ++i) {
                f=std::pow(2,i);
                dxlevels.emplace_back(Matrix2frow ({ (mx(0)+1*dxMatch(0))/f, (mx(1)+1*dxMatch(1))/f}));

                if ( (dxlevels.back().array()<=dxMatch.array()).array().any() )
                        break;
        }

        mxlvl=dxlevels.size();
        Matrix2irow n_edges = ( (mx+1*dxMatch).array()/dxlevels.back().array()+1 ).cast<int>();
        MatrixXXi H1match= computeHitogram2D(Xtarg1,n_edges, Matrix2frow({0,0}), mx+1*dxMatch);

        MatrixXXi H2match = H1match.unaryExpr(
                [](int x) {
                return ((x>0) ? 1 : 0 );
        });

        HLevels.push_back(H2match);
        for(int i=1; i<mxlvl; ++i) {
                auto Hup = HLevels.back();
                auto H=UpsampleMax(Hup,n);
                HLevels.push_back(H);
        }

        mxLVL=int(HLevels.size())-1;
        std::reverse(HLevels.begin(),HLevels.end());



}

BinMatchSol
BinMatch::getmatch(const Eigen::Ref<const MatrixX2f>& Xsrc,const Eigen::Ref<const Eigen::Matrix3f>& H12){
        Eigen::Matrix3f H12mn = H12;
        H12mn.block(0,2,2,0)=H12mn.block(0,2,2,0)-mn_orig.matrix().transpose();
        Matrix2frow t0;
        t0(0)= H12mn(0,2);
        t0(1)= H12mn(1,2);
        Matrix2frow L0=t0-Lmax;
        Matrix2frow L1=t0+Lmax;



        SolBox solbox_init;
        solbox_init.lb = Matrix2frow({0,0});
        solbox_init.dx = dxlevels[0];
        solbox_init.cost=0;
        solbox_init.lvl=0;
        solbox_init.th=0;
        solbox_init.flg=false; //


        int lvl=0;
        auto dx=dxlevels[lvl];
        auto H=HLevels[lvl];

        std::vector<SolBox> qv;
        for(float th=-thmax; th<=thmax+thfineres; th=th+thfineres) {

                Eigen::Matrix2f R ({{std::cos(th), -std::sin(th)},{std::sin(th), std::cos(th)}});
                MatrixX2f XX1 = (R*(Xsrc.transpose())).transpose();
                MatrixX2f XX=((H12mn.block(0,0,2,2))*(XX1.transpose())).transpose();
                Xth[th]=XX;

                SolBox sb = solbox_init;
                sb.th = th;
                qv.push_back(sb);
        }

        BBox bb1;
        bb1.x1=t0(0)-Lmax(0);
        bb1.y1=t0(1)-Lmax(1);
        bb1.x2=t0(0)+Lmax(0);
        bb1.y2=t0(1)+Lmax(1);

        auto cmp = [](SolBox left, SolBox right) {
                           return (left.cost) < (right.cost);
                   };

        // now break the initial box further according to dxBase
        int cnt=0;
        if (dxBase(0)>0) {
                while(1) {
                        cnt=0;
                        for(int i=0; i<qv.size(); ++i) {
                                if ((qv[i].dx.array()>dxBase.array()).array().any()) {
                                        auto qsv = quadSplitSolBox(qv[i]);
                                        for(int j=0; j<qsv.size(); ++j) {
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

    #pragma omp parallel for num_threads(6)
        for(int i=0; i<qv.size(); ++i) {
                qv[i].cost=getPointCost(HLevels[qv[i].lvl],dxlevels[qv[i].lvl],Xth[qv[i].th],qv[i].lb);
                qv[i].flg=true;
        }



        std::priority_queue<SolBox, std::vector<SolBox>, decltype(cmp)> q(cmp,std::move(qv));

        MatrixX2f XX0=((H12mn.block(0,0,2,2))*(Xsrc.transpose())).transpose();
        XX0 = XX0.rowwise()+t0;
        int cost0 = getPointCost(H,dx,XX0,Matrix2frow({0,0}));
        SolBox finalsol;
        while(1) {
                SolBox sb=q.top();
                q.pop();
                if(sb.lvl==mxLVL) {
                        finalsol=sb;
                        break;
                }
                auto qsv = quadSplitSolBox(sb);
                for(int j=0; j<qsv.size(); ++j) {
                        if(SolBoxesIntersect(bb1,qsv[j])) {
                                qsv[j].lvl = sb.lvl+1;
                                qsv[j].cost=getPointCost(HLevels[qsv[j].lvl],dxlevels[qsv[j].lvl],Xth[qsv[j].th],qsv[j].lb);
                                q.push(qsv[j]);
                        }
                }

        }

        auto sb = finalsol;
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
        return bms;
}
