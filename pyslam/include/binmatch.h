#pragma once

#include "base.h"

void takejson(const nlohmann::json& json);

Matrix3f get3fHmat(Matrix4f H,int fixedaxis=2){

}

MatrixXXi
UpsampleMax(const Eigen::Ref<const MatrixXXi>& Hup,int n);


MatrixXXi
computeHitogram2D(const Eigen::Ref<const MatrixX2f>& X,Matrix2irow n_edges, Matrix2frow xmin, Matrix2frow xmax);


int
getPointCost(const Eigen::Ref<const MatrixXXi>& H, const Eigen::Ref<const Matrix2frow>& dx,
             const Eigen::Ref<const Eigen::MatrixX2f>& X,const Eigen::Ref<const Matrix2frow>& Oj);

struct SolBox {
        Matrix2frow lb; // bottom left corner
        Matrix2frow dx; // side length
        int cost=0;
        int lvl=0;
        float th=0;
        bool flg=false; // true only if cost was computed
};
struct BBox {
        float x1,y1;
        float x2,y2;
};

struct BinMatchSol {
        Eigen::Matrix3f H;
        int cost0;
        int cost;
        int lvl;
        int mxLVL;
};

std::ostream& operator<<(std::ostream& os, const SolBox& sb);

std::ostream& operator<<(std::ostream& os, const BBox& sb);

std::ostream& operator << (std::ostream& os, const std::vector<SolBox>& v);

BBox SolBox2BBox(const SolBox& solbox);

std::vector<SolBox> quadSplitSolBox(const SolBox& solbox);

bool SolBoxesIntersect(const SolBox& sb1,const SolBox& sb2);

bool SolBoxesIntersect(const BBox& bb1,const SolBox& sb2);


class BinMatch {
public:
BinMatch(std::string options_);
void setOptions(std::string options_);
void computeHlevels(const Eigen::Ref<const MatrixX2f>& Xtarg);

std::vector<BinMatchSol>
getmatch(const Eigen::Ref<const MatrixX2f>& Xsrc,const Eigen::Ref<const Eigen :: Matrix3f>& H12);

Matrix2frow t0;
Eigen::Matrix3f H12mn;
std::vector<int> levels;
std::vector<MatrixXXi> HLevels;
std::vector<Matrix2frow> dxlevels;
Matrix2frow dxMatch;
Matrix2frow dxBase;
Matrix2frow Lmax;
std::unordered_map<float,MatrixX2f> Xth;
float thmax;
float thfineres;
json options;
Matrix2frow mn_orig;
int mxLVL;

std::vector<SolBox> qvinit;
};
