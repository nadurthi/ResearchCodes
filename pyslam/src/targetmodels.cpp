#include "targetmodels.h"


TargetModel::TargetModel(std::string opt){
        options=json::parse(opt);
}
Eigen::VectorXf TargetModel::propforward(Eigen::VectorXf x){

}


CarModel3D::CarModel3D(std::string opt) : TargetModel(opt){
        options=json::parse(opt);
}

Eigen::VectorXf CarModel3D::propforward(Eigen::VectorXf x){



}

Eigen::VectorXf CarModel3D::randinit(Eigen::VectorXf lb,Eigen::VectorXf ub){



}
