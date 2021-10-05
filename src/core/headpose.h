#ifndef _HEADPOSE_H_
#define _HEADPOSE_H_

#include "net.h"
#include "common.h"
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>

namespace core {
	class HeadPose{
	public:
		HeadPose();
		~HeadPose();
		int LoadModel(const char* root_path);
		// tried use void Predict but appear error : https://stackoverflow.com/questions/8896281/why-am-i-getting-void-value-not-ignored-as-it-ought-to-be/17915388
		// in engine wapper. when in inference::HeadPoseEngine::Predict is funtion type int
		int Predict(const cv::Mat& image, std::vector<FaceInfo>& faces,
                        std::vector<PoseValue>* poses, std::vector<ScaleInfo>* scales_info);

	private:
		ncnn::Net* hopenet;
        bool initialized_;
        const cv::Size inputSize_ = { 224, 224 };
		const float mean[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f}; // image = (image - mean) / std
    	const float norm[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f}; // convert back before transform: ((image * std) + mean)
		void PreProcess(cv::Mat& image, FaceInfo& faceinfo,
                        cv::Mat& input_img, ScaleInfo& scaleinfo);
		void PostProcess(std::vector<float> yaw, std::vector<float> pitch,
                                std::vector<float> roll, PoseValue& pose_value);
	};

	/*
		End of namespace
	*/
}

#endif // !_HEAHPOSE_H_
