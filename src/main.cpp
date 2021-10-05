#include <iostream>
#include "engine.h"
#include <opencv2/opencv.hpp>

using namespace core;
int main(int argc, char** argv)
{
    const char* root_path = argv[1];
    std::vector<FaceInfo> faces;
    std::vector<PoseValue> posevalues;
    std::vector<ScaleInfo> scalesinfo;
    cv::Mat image = cv::imread(argv[2]);
    //
    inference::HeadPoseEngine* engine = new inference::HeadPoseEngine();
    engine->LoadModel(root_path);
    engine->DetectFace(image, &faces);
    double start_estimate = static_cast<double>(cv::getTickCount());
    engine->Predict(image, faces, &posevalues, &scalesinfo);
    double end_estimate = static_cast<double>(cv::getTickCount());
    double time = (end_estimate-start_estimate)/cv::getTickFrequency() * 1000;
    std::cout << "Time Head Pose Estimate: " << time << std::endl;
    //
    int tdx_ = static_cast<int>((scalesinfo[0].x_min + scalesinfo[0].x_max) / 2);
    int tdy_ = static_cast<int>((scalesinfo[0].y_min + scalesinfo[0].y_max) / 2);
    int size_ = static_cast<int>(scalesinfo[0].box_heigh/2);
    float v_yaw = posevalues[0].yaw, v_pitch = posevalues[0].pitch, v_roll = posevalues[0].roll;
    draw_axis(image, v_yaw, v_pitch, v_roll, tdx_, tdy_, size_);
    // cv::imwrite("../images/pose.jpg", image);
    std::cout << faces[0].location_ << std::endl;
    cv::Mat output;
    draw(image, faces, output);
    cv::imwrite("../images/output.jpg", output);
    return 0;
}