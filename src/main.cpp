#include <iostream>
#include "engine.h"
#include <opencv2/opencv.hpp>

using namespace core;
int main(){
    const char* root_path = "../models";
    std::vector<FaceInfo> faces;
    std::vector<HeadInfo> headinfo;
    std::vector<PoseValue> posevalues;
    std::vector<ScaleInfo> scalesinfo;
    cv::Mat image = cv::imread("../images/cr7.jpg");
    //
    inference::HeadPoseEngine* engine = new inference::HeadPoseEngine();
    engine->LoadModel(root_path);
    
    engine->DetectFace(image, &faces);
    // for (int i=0; i<20; i++){
    double start = static_cast<double>(cv::getTickCount());
    engine->Predict(image, faces, &posevalues, &scalesinfo);
    double end = static_cast<double>(cv::getTickCount());
    double time = (end-start)/cv::getTickFrequency() * 1000;
    std::cout << "Time estimate: " << time << std::endl;
    // }
    std::cout << "yaw: " << posevalues[0].yaw << std::endl;
    std::cout << "pitch: " << posevalues[0].pitch << std::endl;
    std::cout << "roll: " << posevalues[0].roll << std::endl;
    //
    int tdx_ = static_cast<int>((scalesinfo[0].x_min + scalesinfo[0].x_max) / 2);
    int tdy_ = static_cast<int>((scalesinfo[0].y_min + scalesinfo[0].y_max) / 2);
    int size_ = static_cast<int>(scalesinfo[0].box_heigh/2);
    float v_yaw = posevalues[0].yaw, v_pitch = posevalues[0].pitch, v_roll = posevalues[0].roll;
    std::cout << "tdx: " << tdx_ << " - tdy: " << tdy_ << std::endl;
    draw_axis(image, v_yaw, v_pitch, v_roll, tdx_, tdy_, size_);
    cv::imwrite("../images/pose.jpg", image);
    std::cout << faces[0].location_ << std::endl;
    cv::Mat output;
    draw(image, faces, output);
    cv::imwrite("../images/o.jpg", output);




    //
    // std::cout << "yaw softmax: " << yaw_value << std::endl;
    // std::cout << "pitch softmax: " << pitch_value << std::endl;
	// std::cout << "roll softmax: " << roll_value << std::endl;
    return 0;
}