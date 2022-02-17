#pragma once
//basic
#include <iostream>
#include <fstream>

//pybind
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>  // py::scoped_interpreter



//stl
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <thread>
#include <map>
#include <unordered_map>
#include <queue>
#include <array>
#include <numeric>
#include <cstddef>

//eigen`
#include <Eigen/Core>
#include <Eigen/Geometry>


//pcl
#include <pcl/registration/gicp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/don.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
// #include <kdtree.h>
// #include <pcl/registration/gicp.h>
#include <pcl/registration/mygicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/console/parse.h>


//misc
#include <nlohmann/json.hpp>
#include <omp.h>


namespace py = pybind11;


using MatrixXbcol = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
using Matrix2frow = Eigen::Matrix<float, 1, 2>;
using Matrix2irow = Eigen::Matrix<int, 1, 2>;
using MatrixXbrow = Eigen::Matrix<bool, 1, Eigen::Dynamic>;
using MatrixXirow = Eigen::Matrix<int, 1, Eigen::Dynamic>;
using MatrixXfrow = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using MatrixXXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixX2f = Eigen::Matrix<float, Eigen::Dynamic, 2>;
using MatrixX3f = Eigen::Matrix<float, Eigen::Dynamic, 3>;
using MatrixX2i = Eigen::Matrix<int, Eigen::Dynamic, 2>;
using MatrixX2d = Eigen::Matrix<double, Eigen::Dynamic, 2>;

using MatrixXXuint16 = Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>;

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector6f = Eigen::Matrix<float, 6, 1>;
using Vector4f = Eigen::Matrix<float, 4, 1>;
using VectorXf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using ArrayXbcol = Eigen::Array<bool, Eigen::Dynamic, 1>;
using Array2frow = Eigen::Array<float, 1, 2>;
using Array2irow = Eigen::Array<int, 1, 2>;
using ArrayXbrow = Eigen::Array<bool, 1, Eigen::Dynamic>;
using ArrayXirow = Eigen::Array<int, 1, Eigen::Dynamic>;
using ArrayXfrow = Eigen::Array<float, 1, Eigen::Dynamic>;
using ArrayXXi = Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>;
using ArrayXXf = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>;
using ArrayX2f = Eigen::Array<float, Eigen::Dynamic, 2>;
using ArrayX2i = Eigen::Array<int, Eigen::Dynamic, 2>;

using json = nlohmann::json;

json
parseOptions(std::string opt);

json
readOptionsFile(std::string file);


template <class myType>
void printmsg(std::string var, myType b) {
 std::cout<< var <<" = " << b << std::endl;
}
