#include <iostream>
#include "engine.h"

namespace inference {
    class HeadPoseEngine::Impl {
        public:
            Impl() {
                detector = new RetinaFace();
                headpose = new HeadPose();
            }

            ~Impl() {
                if (detector) {
                    delete detector;
                    detector = nullptr;
                }

                if (headpose) {
                    delete headpose;
                    headpose = nullptr;
                }
            }

            int LoadModel(const char* root_path) {
                if (detector->LoadModel(root_path) != 0) {
                    std::cout << "load face detecter failed." << std::endl;
                    return 10000;
                }

                if (headpose->LoadModel(root_path) != 0) {
                    std::cout << "load face landmarker failed." << std::endl;
                    return 10000;
                }
                return 0;
            }
            inline int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
                return detector->DetectFace(img_src, faces);
            }
            inline int Predict(const cv::Mat& image, std::vector<FaceInfo>& faces,
                        std::vector<PoseValue>* poses, std::vector<ScaleInfo>* scales_info) {
                return headpose->Predict(image, faces, poses, scales_info);
            }

        private:
            RetinaFace* detector = nullptr;
            HeadPose* headpose = nullptr;


    };

    HeadPoseEngine::HeadPoseEngine() {
        impl_ = new HeadPoseEngine::Impl();
    }

    HeadPoseEngine::~HeadPoseEngine() {
        if (impl_) {
            delete impl_;
            impl_ = nullptr;
        }
    }

    int HeadPoseEngine::LoadModel(const char* root_path) {
        return impl_->LoadModel(root_path);
    }

    int HeadPoseEngine::DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
        return impl_->DetectFace(img_src, faces);
    }

    int HeadPoseEngine::Predict(const cv::Mat& image, std::vector<FaceInfo>& faces,
                        std::vector<PoseValue>* poses, std::vector<ScaleInfo>* scales_info) {
        return impl_->Predict(image, faces, poses, scales_info);
    }
}