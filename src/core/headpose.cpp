#include "headpose.h"
#include <iostream>
#include <math.h>

namespace core {
    // Constructor
	HeadPose::HeadPose(){
		hopenet = new ncnn::Net();
	}
    // Deconstructor
	HeadPose::~HeadPose() {
		if (hopenet) {
			hopenet->clear();
		}
	}
    // load ncnn model into net
	int HeadPose::LoadModel(const char * root_path) {
		std::string fd_param = std::string(root_path) + "/headpose/robust_alpha.param";
		std::string fd_bin = std::string(root_path) + "/headpose/robust_alpha.bin";
		if (hopenet->load_param(fd_param.c_str()) == -1 ||
			hopenet->load_model(fd_bin.c_str()) == -1) {
			std::cout << "load deep head pose model failed." << std::endl;
			return 10000;
		}
        initialized_ = true;
		return 0;
	}
    // Pre processing input
    void HeadPose::PreProcess(cv::Mat& img_cpy, FaceInfo& faceinfo,
                        cv::Mat& input_img, ScaleInfo& scaleinfo)
    {
        /**
         * @faceinfo:
        */
        int img_width = img_cpy.cols;
        int img_height = img_cpy.rows;
        // extract location
        float x_min = faceinfo.location_.tl().x;
        float y_min = faceinfo.location_.tl().y;
        float x_max = faceinfo.location_.br().x;
        float y_max = faceinfo.location_.br().y;
        // scaling bouding box
        float bbox_width = abs(x_max - x_min);
        float bbox_height = abs(y_max - y_min);
        //
        float width_scale = static_cast<float>(2 * bbox_width / 4);
        float height_scale = static_cast<float>(bbox_height / 4);
        //
        float x_min_new =  x_min - width_scale;
        float x_max_new =  x_max + width_scale;
        float y_min_new = y_min - 3 * height_scale;
        float y_max_new = y_max + height_scale;
        // Get bbox to crop image
        float threshold = 0.0, tl_x, tl_y, br_x, br_y;
        tl_x = std::max(x_min_new, threshold);
        tl_y = std::max(y_min_new, threshold);
        float x_max_t = std::min(static_cast<float>(img_width), x_max_new);
        float y_max_t = std::min(static_cast<float>(img_height), y_max_new);
        br_x = x_max - tl_x; // subtract margin coords to x, y
        br_y = y_max - tl_y;
        // Get bbox values
        scaleinfo.box_heigh = bbox_height;
        scaleinfo.box_width = bbox_width;
        scaleinfo.x_max=x_max_t;
        scaleinfo.y_max=y_max_t;
        scaleinfo.x_min=tl_x;
        scaleinfo.y_min=tl_y;
        //
        cv::Rect roi = cv::Rect(tl_x, tl_y, br_x, br_y);
        //
        // input_img = img_cpy(cv::Range(roi.y, roi.height), cv::Range(roi.x, roi.width));
        input_img = img_cpy(roi);
    }
    // Post processing network output
    void HeadPose::PostProcess(std::vector<float> yaw, std::vector<float> pitch,
                                std::vector<float> roll, PoseValue& pose_value)
    {
        /**
         * @heads_info: 
         * @post_values:
        */
			std::vector<float> yaw_predicted, pitch_predicted, roll_predicted;
			softmax(yaw, yaw_predicted);
			softmax(pitch, pitch_predicted);
			softmax(roll, roll_predicted);
			// Get continuous predictions in degrees.
			float yaw_value=0.0, pitch_value=0.0, roll_value=0.0;
			for(int index=0; index<66; index++){
				yaw_value += yaw_predicted[index] * index;
				pitch_value += pitch_predicted[index] * index;
				roll_value += roll_predicted[index] * index;
			}
			pose_value.yaw = yaw_value*3-99;
			pose_value.pitch = pitch_value*3-99;
			pose_value.roll = roll_value*3-99;
	}
    // predict yaw, pitch, roll from image
    int HeadPose::Predict(const cv::Mat& image, std::vector<FaceInfo>& faces,
                        std::vector<PoseValue>* poses, std::vector<ScaleInfo>* scales_info)
    {
        //
        /**
         * 
         * @faces:
         * @poses:
         * 
        */
		poses->clear(); // free
        scales_info->clear();
        cv::Mat img_cpy = image.clone();

		if (!initialized_) {
			assert ("deep head pose model uninitialized.");
		}
		if (image.empty()) {
			assert("input image empty.");
		}
        // execute
        for(int i=0; i<faces.size(); i++){
            //
            ScaleInfo scale_info;
            PoseValue pose_value;
            // Preprocess input state
            FaceInfo faceinfo = faces[i];
            
            cv::Mat input_img;
            HeadPose::PreProcess(img_cpy, faceinfo, input_img, scale_info);
            cv::imwrite("../images/crop.jpg", input_img);
            // prepare ncnn extractor
            ncnn::Extractor ex = hopenet->create_extractor();
            ncnn::Mat in = ncnn::Mat::from_pixels_resize(input_img.data,
                ncnn::Mat::PIXEL_BGR2RGB, input_img.cols, input_img.rows, inputSize_.width, inputSize_.height);
            ex.input("input.1", in);
            //
            std::string yaw_layer_name = "511";
            std::string pitch_layer_name = "510";
            std::string roll_layer_name = "509";

            ncnn::Mat yaw_mat, pitch_mat, roll_mat;
            ex.extract(yaw_layer_name.c_str(), yaw_mat);
            ex.extract(pitch_layer_name.c_str(), pitch_mat);
            ex.extract(roll_layer_name.c_str(), roll_mat);
            // ncnn mat to array
            std::vector<float> yaw, pitch, roll;
            HeadInfo head_info;
            for (int j = 0; j < 66; j++)
            {
                yaw.push_back(yaw_mat[j]);
                pitch.push_back(pitch_mat[j]);
                roll.push_back(roll_mat[j]);
            }
            // Post process output state
            HeadPose::PostProcess(yaw, pitch, roll, pose_value);
            //
            scales_info->push_back(scale_info);
            poses->push_back(pose_value);
        }

        return 0;
    }

    /*
        End of namespace
    */

}