#pragma once
#include "base.h"

using Vector6d = Eigen::Matrix<double, 6, 1>;
using json = nlohmann::json;

namespace py = pybind11;

std::map<std::string, int> convert_dict_to_map(py::dict dictionary);


int add(int i,int j,py::dict dict,std::string c);

void print4x4Matrix (const Eigen::Matrix4f & matrix);

std::vector<std::pair<std::string,Eigen::MatrixXf> >  registrations(const Eigen::Ref<const Eigen::MatrixXf> &big,const Eigen::Ref<const Eigen::MatrixXf> &small,std::string c);
