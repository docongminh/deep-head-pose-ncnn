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

    cv::putText(image, "yaw: "+std::to_string(v_yaw).substr(0, std::to_string(v_yaw).find(".")+3), cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0, 255), 2);
    cv::putText(image, "pitch: "+std::to_string(v_pitch).substr(0, std::to_string(v_pitch).find(".")+3), cv::Point(50, 80), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 255, 0), 2);
    cv::putText(image, "roll: "+std::to_string(v_roll).substr(0, std::to_string(v_roll).find(".")+3), cv::Point(50, 110), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 2);
    //
    std::cout << "yaw: " << v_yaw << std::endl;
    std::cout << "pitch: " << v_pitch << std::endl;
    std::cout << "roll: " << v_roll << std::endl;
    // cv::imwrite("../images/pose.jpg", image);
    std::cout << faces[0].location_ << std::endl;
    cv::Mat output;
    draw(image, faces, output);

    cv::imwrite("../images/output.jpg", output);
    return 0;
}