#pragma once

#include "base.h"
#include "targetmodels.h"






class PF {
public:
PF(std::string opt);

void propforward();

void measUpdt(Eigen::VectorXf likelihood );

void renormalize();

void bootstrapresample();
// float Neff();

json options;
Eigen::MatrixXf X;
Eigen::VectorXf W;
TargetModel model;
};
