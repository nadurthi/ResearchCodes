///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/*********************************************************************
 ** This sample demonstrates how to capture a live 3D point cloud   **
 ** with the ZED SDK and display the result in an OpenGL window.    **
 *********************************************************************/



#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>
#include "cv-helpers.hpp"
#include "myhelper.hpp"

// #include <iostream>
// #include <algorithm>            // std::min, std::max
// #include <pcl/point_types.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/io/ply_io.h>
// #include <pcl/filters/passthrough.h>
// #include <string>
// #include <chrono>
// #include <ctime>   // localtime
// #include <termios.h>
// #include <iomanip> // put_time
// #include <time.h> 
// #include <boost/filesystem.hpp>
// #include <signal.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include <unistd.h>
// #include <unistd.h>
// #include <iostream>
// #include <cstdlib>
// #include <signal.h>
// #include <Eigen/Dense>
// #include <thread>         // std::thread
// #include <queue>          // std::queue
// #include <mutex>
// #include <memory>
// #include <fstream>              // File IO
// #include <iostream>             // Terminal IO
// #include <sstream>              // Stringstreams
// #include <opencv2/opencv.hpp>
// #include "cv-helpers.hpp"
// #include <map>
// #include <utility>
// #include <vector>

using namespace boost::filesystem;
using namespace std;
using namespace cv;
using Eigen::MatrixXd;

int EXIT_FLAG;
// -----------  Ctrl+C capture ----------------------------

// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) {
   cout << "Caught signal " << signum << endl;
   // Terminate program
   EXIT_FLAG = 1;
}

// -----------  EIGEN ----------------------------

MatrixXd timeVector(100000,1);

int cnt=0;


// -----------  PCL PCD ----------------------------


pcl_ptr points_to_pcl2(const rs2::points& points)
{
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }
    // std:string filename = pcdfolder+std::string("/test_pcd_")+std::to_string(ind)+std::string(".pcd");
    // // pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    // pcl::io::savePCDFileASCII (filename.c_str(), *cloud);

    return cloud;
}




class ThreadSaver{
public:
    ThreadSaver(std::string pcdfolderr,std::string colorfolderr,std::string depthfolderr){
        pcdfolder=pcdfolderr;
        colorfolder=colorfolderr;
        depthfolder=depthfolderr;
        pcdind=0;
        colind=0;
        depind=0;
    }
    std::string pcdfolder;
    std::string colorfolder;
    std::string depthfolder;

    std::queue< pcl_ptr > myqueue;
    std::queue< cv::Mat > colqueue;
    std::queue< cv::Mat > depqueue;

    std::mutex q_mutex_cld,q_mutex_color,q_mutex_depth;
    
    int pcdind;
    int colind;
    int depind;

    void pushpoints(const rs2::points& points){
        auto cloudptr = points_to_pcl2(points);
        q_mutex_cld.lock();
        myqueue.push(cloudptr);
        q_mutex_cld.unlock();

    }
    void pushcolor(const cv::Mat& color){
        
        q_mutex_color.lock();
        colqueue.push(color.clone());
        q_mutex_color.unlock();

    }
    void pushdepth(const cv::Mat& depth){
        
        q_mutex_depth.lock();
        depqueue.push(depth.clone());
        q_mutex_depth.unlock();

    }

    void dummy(){
        std::cout<<"DUMMY to file the pcd data "<<std::endl;
    }
    void processSavingCloudPtr(){
        while(1){
            if (myqueue.empty() && EXIT_FLAG==0){
                std::this_thread::sleep_for (std::chrono::seconds(1));
                continue;
            }
            q_mutex_cld.lock();
            auto cloudptr = myqueue.front();
            q_mutex_cld.unlock();

            std:string filename = pcdfolder+std::string("/pcd_")+std::to_string(pcdind)+std::string(".bin");
            pcl::io::savePCDFileBinary (filename.c_str(), *cloudptr);
            pcdind++;
            
            std::cout<<"Writing to file the pcd data "<<std::endl;
            
            
            myqueue.pop();

            if (myqueue.empty() && EXIT_FLAG==1){
                break;
            }
        }
    }
    void processSavingColor(){
        while(1){
            if (colqueue.empty() && EXIT_FLAG==0){
                std::this_thread::sleep_for (std::chrono::seconds(1));
                continue;
            }
            q_mutex_color.lock();
            auto color_mat = colqueue.front();
            q_mutex_color.unlock();

            std::string ffc = colorfolder+std::string("/color_")+std::to_string(colind)+std::string(".png");
            imwrite(ffc.c_str(),color_mat);
            colind++;
            
            std::cout<<"Writing to file the color data "<<std::endl;
            
            
            colqueue.pop();

            if (colqueue.empty() && EXIT_FLAG==1){
                break;
            }
        }
    }

    void processSavingDepth(){
        while(1){
            if (depqueue.empty() && EXIT_FLAG==0){
                std::this_thread::sleep_for (std::chrono::seconds(1));
                continue;
            }
            q_mutex_depth.lock();
            auto depth_mat = depqueue.front();
            q_mutex_depth.unlock();

            
            if (depind ==0) {
                std::string ffd = depthfolder+std::string("/depthCV_64F_")+std::to_string(depind)+std::string(".yml");
                FileStorage fs(ffd.c_str(), FileStorage::WRITE);
                fs << "depth_mat" << depth_mat;
                fs.release();  
            }
            
            std::string ffd = depthfolder+std::string("/depthCV_64F_")+std::to_string(depind)+std::string(".bin");
            write_binary(ffd.c_str(),depth_mat);

            depind++;
            
            std::cout<<"Writing to file the depth data "<<std::endl;
            
            
            depqueue.pop();

            if (depqueue.empty() && EXIT_FLAG==1){
                break;
            }
        }
    }

};

void saveInstrinsics(rs2_intrinsics intrinsics,std::string cam,std::string mainsessionfolder){
    std::ofstream myfile;
    std::string int_file = mainsessionfolder+std::string("/intrinsics_")+cam;
    myfile.open (int_file.c_str());

    auto principal_point = std::make_pair(intrinsics.ppx, intrinsics.ppy);
    auto focal_length = std::make_pair(intrinsics.fx, intrinsics.fy);
    rs2_distortion model = intrinsics.model;

    myfile << "Principal Point         : " << principal_point.first << ", " << principal_point.second << "\n";
    myfile << "Focal Length            : " << focal_length.first << ", " << focal_length.second << "\n";
    myfile << "Distortion Model        : " << model << "\n";
    myfile << "Distortion Coefficients : [" << intrinsics.coeffs[0] << "," << intrinsics.coeffs[1] << "," <<
    intrinsics.coeffs[2] << "," << intrinsics.coeffs[3] << "," << intrinsics.coeffs[4] << "]" << "\n";

    myfile.close();
}

void saveExtrinsics(rs2_extrinsics extrinsics,std::string cam,std::string mainsessionfolder){
    std::ofstream myfile;
    std::string int_file = mainsessionfolder+std::string("/extrinsics_")+cam;
    myfile.open (int_file.c_str());

    myfile << "Translation Vector : [" << extrinsics.translation[0] << "," << extrinsics.translation[1] << "," << extrinsics.translation[2] << "]\n";
            myfile << "Rotation Matrix row0 : [" << extrinsics.rotation[0] << "," << extrinsics.rotation[3] << "," << extrinsics.rotation[6] << "]\n";
            myfile << "Rotation Matrix row1 : [" << extrinsics.rotation[1] << "," << extrinsics.rotation[4] << "," << extrinsics.rotation[7] << "]\n";
            myfile << "Rotation Matrix row2 : [" << extrinsics.rotation[2] << "," << extrinsics.rotation[5] << "," << extrinsics.rotation[8] << "]" << "\n";

    myfile.close();
}



int main(int argc, char **argv) try {
    
    signal(SIGINT, signal_callback_handler);
    EXIT_FLAG = 0;

    

    rs2::context ctx;

    std::cout << "hello from librealsense - " << RS2_API_VERSION_STR << std::endl;
    std::cout << "You have " << ctx.query_devices().size() << " RealSense devices connected" << std::endl;

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;

    // Create a pipeline to easily configure and start the camera
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipe.start(cfg);

    


    // rs2::align align_to_depth(RS2_STREAM_DEPTH);
    // rs2::align align_to_color(RS2_STREAM_COLOR);

    // Define colorizer and align processing-blocks
    rs2::colorizer colorize;
    rs2::align align_to_color(RS2_STREAM_COLOR);

    // Start the camera
    // rs2::pipeline pipe;
    // pipe.start();

    std::string nowtimestr = return_current_time_and_date();

    auto mainsessionfolder = std::string("RealSenseSession_")+nowtimestr;
    if(boost::filesystem::create_directory(mainsessionfolder))
    {
        std::cerr<< "Directory Created: "<<mainsessionfolder<<std::endl;
    }
    auto pcdfolder = mainsessionfolder+std::string("/pcd");
    if(boost::filesystem::create_directory(pcdfolder))
    {
        std::cerr<< "Directory Created: "<<pcdfolder<<std::endl;
    }
    auto colorfolder = mainsessionfolder+std::string("/color");
    if(boost::filesystem::create_directory(colorfolder))
    {
        std::cerr<< "Directory Created: "<<colorfolder<<std::endl;
    }
    auto depthfolder = mainsessionfolder+std::string("/depth");
    if(boost::filesystem::create_directory(depthfolder))
    {
        std::cerr<< "Directory Created: "<<depthfolder<<std::endl;
    }

    auto timevecfilename = mainsessionfolder+std::string("/timesteps.csv");

    std::cout << "mainsessionfolder = "<<mainsessionfolder<<std::endl;
    std::cout << "pcdfolder = "<<pcdfolder<<std::endl;
    std::cout << "colorfolder = "<<colorfolder<<std::endl;
    std::cout << "depthfolder = "<<depthfolder<<std::endl;

    // - Get intrinsics and extrinsics------------------------------
    auto const depthstream = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto const colorstream = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

    rs2_intrinsics const color_intrinsics = colorstream.get_intrinsics();
    rs2_intrinsics const depth_intrinsics = depthstream.get_intrinsics();

    saveInstrinsics(color_intrinsics,"color",mainsessionfolder);
    saveInstrinsics(depth_intrinsics,"depth",mainsessionfolder);

    rs2_extrinsics extrinsics_depth_to_color = depthstream.get_extrinsics_to(colorstream);
    rs2_extrinsics extrinsics_color_to_depth = colorstream.get_extrinsics_to(depthstream);

    saveExtrinsics(extrinsics_depth_to_color,"depth_to_color",mainsessionfolder);
    saveExtrinsics(extrinsics_color_to_depth,"color_to_depth",mainsessionfolder);


    // - Set savinf threads  ------------------------------
    ThreadSaver thdsaver(pcdfolder,colorfolder,depthfolder);
    std::thread thdsaver_points (&ThreadSaver::processSavingCloudPtr,&thdsaver); 
    std::thread thdsaver_color (&ThreadSaver::processSavingColor,&thdsaver); 
    std::thread thdsaver_depth (&ThreadSaver::processSavingDepth,&thdsaver); 


    // Capture 30 frames to give autoexposure, etc. a chance to settle
    for (auto i = 0; i < 30; ++i) {
        pipe.wait_for_frames();       
    }

    int cind=0;
    // int dind=0;

    while (true)
    {   
        

        // Block program until frames arrive
        rs2::frameset frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);

        double fractional_seconds_since_epoch
        = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now().time_since_epoch()).count();
        
        timeVector(cnt++,0) = (double)fractional_seconds_since_epoch;

        printf("Timestep : %f\n",fractional_seconds_since_epoch);

        // Try to get a frame of a depth image
        rs2::depth_frame depth = frames.get_depth_frame();
        cv::Mat depth_mat = depth_frame_to_meters(depth);
        
        auto color_mat = frame_to_mat(frames.get_color_frame());
        auto points = pc.calculate(depth);

        // std::string ffd = depthfolder+std::string("/depthCV_64F_")+std::to_string(dind)+std::string(".yml");
        // FileStorage fs(ffd.c_str(), FileStorage::WRITE);
        // fs << "depth_mat" << depth_mat;
        // fs.release();  
        // dind++;

        

        // Mat color(Size(960, 540), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        // std::string ffc = colorfolder+std::string("/color_")+std::to_string(cind)+std::string(".png");
        // imwrite(ffc.c_str(),color_mat);
        // cind++;


        // namedWindow("Display Image", WINDOW_AUTOSIZE );
        // imshow("Display Image", color);

        // auto infrared = frames.get_infrared_frame();

        // Generate the pointcloud and texture mappings
        

        // double t0 = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count();
        

        thdsaver.pushpoints(points);
        thdsaver.pushcolor(color_mat);
        thdsaver.pushdepth(depth_mat);
        // auto pcl_points = points_to_pcl(pcdfolder,points,++ind);

        // double tf = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count();

        // std::cout<<"time to write pcd = "<< tf-t0 << std::endl;

        // Get the depth frame's dimensions
        auto width = depth.get_width();
        auto height = depth.get_height();

        // Query the distance from the camera to the object in the center of the image
        float dist_to_center = depth.get_distance(width / 2, height / 2);

        // Print the distance
        std::cout<<std::endl << "The camera is facing an object " << dist_to_center << " meters away "<<std::endl ;
        if(EXIT_FLAG==1)
            break;
    }

    cout<<" waiting for joining thread"<<std::endl;
    thdsaver_points.join();
    thdsaver_color.join();
    thdsaver_depth.join();
    cout<<" done with joining thread"<<std::endl;

    writeToCSVfile(timevecfilename, timeVector);

    return EXIT_SUCCESS;

}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}


// catch (const rs2::error & e)
// {
//     std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
//     return EXIT_FAILURE;
// }
// catch (const std::exception& e)
// {
//     std::cerr << e.what() << std::endl;
//     return EXIT_FAILURE;
// }


