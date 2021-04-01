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

// ZED includes
#include <sl/Camera.hpp>
// #include "GLViewer.hpp"
#include "myhelper.hpp"

#include <iostream>
// Using std and sl namespaces
using namespace boost::filesystem;
using namespace std;
using namespace cv;
using namespace sl;
using Eigen::MatrixXd;


int EXIT_FLAG;
// ------------ ZED -------------
// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}
cv::Mat slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

// -----------  Ctrl+C capture ----------------------------

// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) {
   cout << "Caught signal " << signum << endl;
   // Terminate program
   EXIT_FLAG = 1;
}

// -----------  EIGEN ----------------------------

MatrixXd timeVector(1000000,1);

int cnt=0;

class ThreadSaver{
public:
    ThreadSaver(std::string colorfolderr,std::string depthfolderr,std::string dispfolderr){

        colorfolder=colorfolderr;
        depthfolder=depthfolderr;
        dispfolder=dispfolderr;

        colind=0;
        depind=0;
        dispind=0;
    }

    std::string colorfolder;
    std::string depthfolder;
    std::string dispfolder;

    std::queue< cv::Mat > colqueue;
    std::queue< cv::Mat > depqueue;
    std::queue< cv::Mat > dispqueue;

    std::mutex q_mutex_color,q_mutex_depth,q_mutex_disp;
    

    int colind;
    int depind;
    int dispind;

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
    void pushdisp(const cv::Mat& disp){
        
        q_mutex_disp.lock();
        dispqueue.push(disp.clone());
        q_mutex_disp.unlock();

    }

    void dummy(){
        std::cout<<"DUMMY to file the pcd data "<<std::endl;
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

            if (colind==0){
            std::string ffc = colorfolder+std::string("/color_")+std::to_string(colind)+std::string(".png");
            imwrite(ffc.c_str(),color_mat);
            }

            std::string ffc = colorfolder+std::string("/color_")+std::to_string(colind)+std::string(".bin");
            write_binary(ffc.c_str(),color_mat);

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
                std::string ffd = depthfolder+std::string("/depth_")+std::to_string(depind)+std::string(".yml");
                FileStorage fs(ffd.c_str(), FileStorage::WRITE);
                fs << "depth_mat" << depth_mat;
                fs.release();  
            }
            
            std::string ffd = depthfolder+std::string("/depth_")+std::to_string(depind)+std::string(".bin");
            write_binary(ffd.c_str(),depth_mat);

            depind++;
            
            std::cout<<"Writing to file the depth data "<<std::endl;
            
            
            depqueue.pop();

            if (depqueue.empty() && EXIT_FLAG==1){
                break;
            }
        }
    }
    void processSavingDisparity(){
        while(1){
            if (dispqueue.empty() && EXIT_FLAG==0){
                std::this_thread::sleep_for (std::chrono::seconds(1));
                continue;
            }
            q_mutex_disp.lock();
            auto disp_mat = dispqueue.front();
            q_mutex_disp.unlock();

            
            if (dispind ==0) {
                std::string ffd = dispfolder+std::string("/disp_")+std::to_string(dispind)+std::string(".yml");
                FileStorage fs(ffd.c_str(), FileStorage::WRITE);
                fs << "disp_mat" << disp_mat;
                fs.release();  
            }
            
            std::string ffd = dispfolder+std::string("/disp_")+std::to_string(dispind)+std::string(".bin");
            write_binary(ffd.c_str(),disp_mat);

            dispind++;
            
            std::cout<<"Writing to file the disp data "<<std::endl;
            
            
            dispqueue.pop();

            if (dispqueue.empty() && EXIT_FLAG==1){
                break;
            }
        }
    }
};

int main(int argc, char **argv) {
    
    signal(SIGINT, signal_callback_handler);
    EXIT_FLAG = 0;

    Camera zed;
    // Set configuration parameters for the ZED
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::HD720;
    init_parameters.depth_mode = DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.depth_minimum_distance = 0.30;
    init_parameters.depth_maximum_distance=20;
    init_parameters.camera_fps = 30;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        std::cout<<"Camera Open "<< returned_state<< " Exit program.";
        return EXIT_FAILURE;
    }

    auto camera_config = zed.getCameraInformation().camera_configuration;

    // Point cloud viewer
    // GLViewer viewer;
    // // Initialize point cloud viewer 
    // GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam);
    // if (errgl != GLEW_OK) {
    //     print("Error OpenGL: " + std::string((char*)glewGetErrorString(errgl)));
    //     return EXIT_FAILURE;
    // }
    std::string nowtimestr = return_current_time_and_date();

    auto mainsessionfolder = std::string("ZedSession_")+nowtimestr;
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
    auto dispfolder = mainsessionfolder+std::string("/disp");
    if(boost::filesystem::create_directory(dispfolder))
    {
        std::cerr<< "Directory Created: "<<dispfolder<<std::endl;
    }
    auto timevecfilename = mainsessionfolder+std::string("/timesteps.csv");

    // - Set savinf threads  ------------------------------
    ThreadSaver thdsaver(colorfolder,depthfolder,dispfolder);
    std::thread thdsaver_color (&ThreadSaver::processSavingColor,&thdsaver); 
    std::thread thdsaver_depth (&ThreadSaver::processSavingDepth,&thdsaver); 
    std::thread thdsaver_disp (&ThreadSaver::processSavingDisparity,&thdsaver); 

    RuntimeParameters runParameters;
    // Setting the depth confidence parameters
    runParameters.confidence_threshold = 50;
    runParameters.texture_confidence_threshold = 100;

    Resolution image_size = zed.getCameraInformation().camera_resolution;
    int new_width = image_size.width / 2;
    int new_height = image_size.height / 2;

    // Allocation of 4 channels of float on GPU
    // sl::Mat point_cloud(camera_config.resolution, MAT_TYPE::F32_C4, MEM::GPU);
    sl::Mat image;
    sl::Mat depth_map;
    sl::Mat disp;

    int cind=0;
    // Main Loop
    while (1) {        
        // Check that a new image is successfully acquired
        if (zed.grab(runParameters) == ERROR_CODE::SUCCESS) {
            double fractional_seconds_since_epoch
            = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::system_clock::now().time_since_epoch()).count();
            
            timeVector(cnt++,0) = (double)fractional_seconds_since_epoch;
            printf("Timestep : %f\n",fractional_seconds_since_epoch);
            // retrieve the current 3D coloread point cloud in GPU
            zed.retrieveImage(image, VIEW::LEFT); // Retrieve left image
            zed.retrieveMeasure(depth_map, MEASURE::DEPTH); // Retrieve depth
            zed.retrieveMeasure(disp, MEASURE::DISPARITY); // Retrieve depth

            cv::Mat depth_image_ocv = slMat2cvMat(depth_map);
            cv::Mat color_image_ocv = slMat2cvMat(image);
            cv::Mat disp_image_ocv = slMat2cvMat(disp);

            // std::string fcol = colorfolder+"/color_"+std::to_string(cind)+".png";
            // image.write(fcol.c_str());
            // cind++;

            thdsaver.pushcolor(color_image_ocv);
            thdsaver.pushdepth(depth_image_ocv);
            thdsaver.pushdisp(disp_image_ocv);
        }

        if(EXIT_FLAG==1)
            break;
    }
    // free allocated memory before closing the ZED
    // point_cloud.free();
    cout<<" waiting for joining thread"<<std::endl;
    thdsaver_color.join();
    thdsaver_depth.join();
    thdsaver_disp.join();
    cout<<" done with joining thread"<<std::endl;

    std::cout << "writing time steps "<< std::endl;
    writeToCSVfile(timevecfilename, timeVector);

    // close the ZED
    zed.close();

    return EXIT_SUCCESS;
}


