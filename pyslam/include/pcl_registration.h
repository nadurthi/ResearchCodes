#pragma once

#include <pybind11/eigen.h>
#include <iostream>
#include <thread>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <map>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <string>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace py = pybind11;

std::map<std::string, int> convert_dict_to_map(py::dict dictionary);


int add(int i,int j,py::dict dict,std::string c);

void print4x4Matrix (const Eigen::Matrix4f & matrix);

std::vector<Eigen::MatrixXf>  registrations(const Eigen::Ref<const Eigen::MatrixXf> &big,const Eigen::Ref<const Eigen::MatrixXf> &small,py::dict dict);
