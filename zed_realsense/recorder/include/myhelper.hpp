#pragma once

#include <iostream>
#include <algorithm>            // std::min, std::max
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>
#include <string>
#include <chrono>
#include <ctime>   // localtime
#include <termios.h>
#include <iomanip> // put_time
#include <time.h> 
#include <boost/filesystem.hpp>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <signal.h>
#include <Eigen/Dense>
#include <thread>         // std::thread
#include <queue>          // std::queue
#include <mutex>
#include <memory>
#include <fstream>              // File IO
#include <iostream>             // Terminal IO
#include <sstream>              // Stringstreams
#include <opencv2/opencv.hpp>
// #include "cv-helpers.hpp"
#include <map>
#include <utility>
#include <vector>

using namespace boost::filesystem;
using namespace std;
using namespace cv;
using Eigen::MatrixXd;
using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, MatrixXd matrix)
{
    std::ofstream file(name.c_str(),std::ofstream::out);
    file << matrix.format(CSVFormat);
 }




// -----------  date time string  ----------------------------

std::string return_current_time_and_date()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}







// ------------------ Binary write cv::Mat ------------------------
inline void write_binary(const std::string& filename, const cv::Mat& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if(out.is_open()) {
        int rows=matrix.rows, cols=matrix.cols;
        int nbytes = matrix.elemSize();
		out.write(reinterpret_cast<char*>(&rows), sizeof(int));
        out.write(reinterpret_cast<char*>(&cols), sizeof(int));
        out.write(reinterpret_cast<char*>(&nbytes), sizeof(int));
        out.write(reinterpret_cast<const char*>(matrix.data), rows*cols*static_cast<int>( matrix.elemSize()) );
        out.close();
    }
    else {
        std::cout << "Can not write to file: " << filename << std::endl;
    }
}

inline void read_binary(const std::string& filename, const cv::Mat& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in.is_open()) {
        int rows=0, cols=0,nbytes=0;
        in.read(reinterpret_cast<char*>(&rows),sizeof(int));
        in.read(reinterpret_cast<char*>(&cols),sizeof(int));
        in.read(reinterpret_cast<char*>(&nbytes),sizeof(int));
        matrix.resize(rows, cols);
        in.read(reinterpret_cast<char*>(matrix.data), rows*cols*nbytes );
        in.close();
    }
    else {
        std::cout << "Can not open binary matrix file: " << filename << std::endl;
    }
}
