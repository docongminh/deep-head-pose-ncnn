#ifndef _RETINAFACE_H_
#define _RETINAFACE_H_

#include "net.h"
#include "common.h"

namespace core {
	using ANCHORS = std::vector<cv::Rect>;
	class RetinaFace{
	public:
		RetinaFace();
		~RetinaFace();
		int LoadModel(const char* root_path);
		int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

	private:
		ncnn::Net* retina_net_;
		bool initialized_;
		std::vector<ANCHORS> anchors_generated_;
		const int RPNs_[3] = { 32, 16, 8 };
		const cv::Size inputSize_ = { 300, 300 };
		const float iouThreshold_ = 0.4f;
		const float scoreThreshold_ = 0.8f;

	};

}

#endif // !_RETINAFACE_H_
