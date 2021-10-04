#ifndef _ENGINE_H_
#define _ENGINE_H_

#include <vector>
#include "headpose.h"
#include "retinaface.h"
#include "common.h"
#include "opencv2/core.hpp"

using namespace core;

namespace inference {
    class HeadPoseEngine {
        public:
            HeadPoseEngine();
            ~HeadPoseEngine();
            //
            int LoadModel(const char* root_path);
            //
            int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);
            int Predict(const cv::Mat& image, std::vector<FaceInfo>& faces,
                        std::vector<PoseValue>* poses, std::vector<ScaleInfo>* scales_info);

        private:
            class Impl;
            Impl* impl_;

    };

}

#endif // !_ENGINE_H_