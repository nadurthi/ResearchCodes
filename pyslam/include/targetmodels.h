#pragma once
#include "base.h"

class TargetModel {
public:
TargetModel(std::string opt);
Eigen::VectorXf propforward(Eigen::VectorXf x);
json options;
};

class CarModel3D : public TargetModel {
public:
CarModel3D(std::string opt);
Eigen::VectorXf propforward(Eigen::VectorXf x);
Eigen::VectorXf randinit(Eigen::VectorXf lb,Eigen::VectorXf ub);

int dim;
Eigen::VectorXf Q;
json options;
};
