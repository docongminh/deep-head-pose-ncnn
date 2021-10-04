#ifndef _COMMON_H_
#define _COMMON_H_

#include <vector>
#include <string>
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>

namespace core {
    const int threads_num = 2;
    
    struct FaceInfo {
        cv::Rect location_;     // x, y, width, height
        float score_;
        float keypoints_[10];
        bool mask_;
    };

    struct HeadInfo {
        std::vector<float> yaw;
        std::vector<float> pitch;
        std::vector<float> roll;
    };

    struct PoseValue {
        float yaw;
        float pitch;
        float roll;
    };

    struct ScaleInfo {
        float x_min, x_max, y_min, y_max, box_heigh, box_width;
    };

    int RatioAnchors(const cv::Rect & anchor,
        const std::vector<float>& ratios, std::vector<cv::Rect>* anchors);

    int ScaleAnchors(const std::vector<cv::Rect>& ratio_anchors,
        const std::vector<float>& scales, std::vector<cv::Rect>* anchors);

    int GenerateAnchors(const int & base_size,
        const std::vector<float>& ratios, const std::vector<float> scales,
        std::vector<cv::Rect>* anchors);

    float InterRectArea(const cv::Rect & a,
        const cv::Rect & b);

    int ComputeIOU(const cv::Rect & rect1,
        const cv::Rect & rect2, float * iou,
        const std::string& type = "UNION");

    template <typename T>
    int const NMS(const std::vector<T>& inputs, std::vector<T>* result,
        const float& threshold, const std::string& type = "UNION") {
        result->clear();
        if (inputs.size() == 0)
            return -1;
        
        std::vector<T> inputs_tmp;
        inputs_tmp.assign(inputs.begin(), inputs.end());
        std::sort(inputs_tmp.begin(), inputs_tmp.end(),
        [](const T& a, const T& b) {
            return a.score_ > b.score_;
        });

        std::vector<int> indexes(inputs_tmp.size());

        for (int i = 0; i < indexes.size(); i++) {
            indexes[i] = i;
        }

        while (indexes.size() > 0) {
            int good_idx = indexes[0];
            result->push_back(inputs_tmp[good_idx]);
            std::vector<int> tmp_indexes = indexes;
            indexes.clear();
            for (int i = 1; i < tmp_indexes.size(); i++) {
                int tmp_i = tmp_indexes[i];
                float iou = 0.0f;
                ComputeIOU(inputs_tmp[good_idx].location_, inputs_tmp[tmp_i].location_, &iou, type);
                if (iou <= threshold) {
                    indexes.push_back(tmp_i);
                }
            }
        }
        return 0;
    }
    void EnlargeRect(const float& scale, cv::Rect* rect);
    void RectifyRect(cv::Rect* rect);
    void draw(cv::Mat img, std::vector<FaceInfo> face_info, cv::Mat& image);
    void draw_axis(cv::Mat& img, float yaw, float pitch, float roll, int tdx, int tdy, int size);
    void softmax(std::vector<float> &input, std::vector<float>& output);
    

    /*
        End of namespace
    */
}

#endif // !_COMMON_H_