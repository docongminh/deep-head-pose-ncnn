#include "common.h"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <assert.h>

namespace core {
	int RatioAnchors(const cv::Rect & anchor,
		const std::vector<float>& ratios, 
		std::vector<cv::Rect>* anchors) {
		anchors->clear();
		cv::Point center = cv::Point(anchor.x + (anchor.width - 1) * 0.5f,
			anchor.y + (anchor.height - 1) * 0.5f);
		float anchor_size = anchor.width * anchor.height;
		for (int i = 0; i < static_cast<int>(ratios.size()); ++i) {
			float ratio = ratios.at(i);
			float anchor_size_ratio = anchor_size / ratio;
			float curr_anchor_width = std::sqrt(anchor_size_ratio);
			float curr_anchor_height = curr_anchor_width * ratio;
			float curr_x = center.x - (curr_anchor_width - 1)* 0.5f;
			float curr_y = center.y - (curr_anchor_height - 1)* 0.5f;

			cv::Rect curr_anchor = cv::Rect(curr_x, curr_y,
				curr_anchor_width - 1, curr_anchor_height - 1);
			anchors->push_back(curr_anchor);
		}
		return 0;
	}

	int ScaleAnchors(const std::vector<cv::Rect>& ratio_anchors,
		const std::vector<float>& scales, std::vector<cv::Rect>* anchors) {
		anchors->clear();
		for (int i = 0; i < static_cast<int>(ratio_anchors.size()); ++i) {
			cv::Rect anchor = ratio_anchors.at(i);
			cv::Point2f center = cv::Point2f(anchor.x + anchor.width * 0.5f,
				anchor.y + anchor.height * 0.5f);
			for (int j = 0; j < static_cast<int>(scales.size()); ++j) {
				float scale = scales.at(j);
				float curr_width = scale * (anchor.width + 1);
				float curr_height = scale * (anchor.height + 1);
				float curr_x = center.x - curr_width * 0.5f;
				float curr_y = center.y - curr_height * 0.5f;
				cv::Rect curr_anchor = cv::Rect(curr_x, curr_y,
					curr_width, curr_height);
				anchors->push_back(curr_anchor);
			}
		}

		return 0;
	}

	int GenerateAnchors(const int & base_size,
		const std::vector<float>& ratios, 
		const std::vector<float> scales,
		std::vector<cv::Rect>* anchors) {
		anchors->clear();
		cv::Rect anchor = cv::Rect(0, 0, base_size, base_size);
		std::vector<cv::Rect> ratio_anchors;
		RatioAnchors(anchor, ratios, &ratio_anchors);
		ScaleAnchors(ratio_anchors, scales, anchors);
		
		return 0;
	}

	float InterRectArea(const cv::Rect & a, const cv::Rect & b) {
		cv::Point left_top = cv::Point(MAX(a.x, b.x), MAX(a.y, b.y));
		cv::Point right_bottom = cv::Point(MIN(a.br().x, b.br().x), MIN(a.br().y, b.br().y));
		cv::Point diff = right_bottom - left_top;
		return (MAX(diff.x + 1, 0) * MAX(diff.y + 1, 0));
	}

	int ComputeIOU(const cv::Rect & rect1,
		const cv::Rect & rect2, float * iou,
		const std::string& type) {

		float inter_area = InterRectArea(rect1, rect2);
		if (type == "UNION") {
			*iou = inter_area / (rect1.area() + rect2.area() - inter_area);
		}
		else {
			*iou = inter_area / MIN(rect1.area(), rect2.area());
		}

		return 0;
	}


	void EnlargeRect(const float& scale, cv::Rect* rect) {
		float offset_x = (scale - 1.f) / 2.f * rect->width;
		float offset_y = (scale - 1.f) / 2.f * rect->height;
		rect->x -= offset_x;
		rect->y -= offset_y;
		rect->width = scale * rect->width;
		rect->height = scale * rect->height;
	}

	void RectifyRect(cv::Rect* rect) {
		int max_side = MAX(rect->width, rect->height);
		int offset_x = (max_side - rect->width) / 2;
		int offset_y = (max_side - rect->height) / 2;

		rect->x -= offset_x;
		rect->y -= offset_y;
		rect->width = max_side;
		rect->height = max_side;    
	}

	void draw(cv::Mat img, std::vector<FaceInfo> face_info, cv::Mat& image)
	{
		for(int i = 0; i<face_info.size(); i++)
		{
			cv::rectangle(img, face_info.at(i).location_, cv::Scalar(0, 255, 0), 2);
			// for (int num = 0; num < 5; ++num) {
			// 	cv::Point curr_pt = cv::Point(face_info.at(i).keypoints_[num],
			// 									face_info.at(i).keypoints_[num + 5]);
			// 	cv::circle(img, curr_pt, 2, cv::Scalar(255, 0, 255), 2);
			// }
		}
		image = img;
	}
	void softmax(std::vector<float> &input, std::vector<float>& output) {

		int i;
		float m, sum, constant;

		m = -INFINITY;
		for (i = 0; i < input.size(); ++i) {
			if (m < input[i]) {
				m = input[i];
			}
		}

		sum = 0.0;
		for (i = 0; i < input.size(); ++i) {
			sum += exp(input[i] - m);
		}

		constant = m + log(sum);
		for (i = 0; i < input.size(); ++i) {
			output.push_back(exp(input[i] - constant));
		}

	}
	//
	void draw_axis(cv::Mat& img, float yaw, float pitch, float roll, int tdx, int tdy, int size){

		float pitch_n = pitch * M_PI / 180;
		float yaw_n = -(yaw * M_PI / 180);
		float roll_ = roll * M_PI / 180;

		// X-Axis pointing to right. drawn in red
		float x1 = size * (cos(yaw) * cos(roll)) + tdx;
		float y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy;

		// Y-Axis | drawn in green
		//        v
		float x2 = size * (-cos(yaw) * sin(roll)) + tdx;
		float y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy;

		// Z-Axis (out of the screen) drawn in blue
		float x3 = size * (sin(yaw)) + tdx;
		float y3 = size * (-cos(yaw) * sin(pitch)) + tdy;
		//
		cv::Point origin_point(tdx, tdy);
		cv::Point red(static_cast<int>(x1), static_cast<int>(y1));
		cv::Point green(static_cast<int>(x2), static_cast<int>(y2));
		cv::Point blue(static_cast<int>(x3), static_cast<int>(y3));
		//
		cv::line(img, origin_point, red, cv::Scalar(0, 0, 255), 3);
		cv::line(img, origin_point, green, cv::Scalar(0, 255, 0), 3);
		cv::line(img, origin_point, blue, cv::Scalar(255, 0, 0), 2); 
	}
	//

	/*
		End of namespace
	*/
}
