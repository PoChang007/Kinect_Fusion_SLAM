#ifndef SYSTEM_UTILITY_H_
#define SYSTEM_UTILITY_H_

#include <thread>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

namespace Kinfu
{
    class SystemUtility
    {
    public:
        SystemUtility();
        ~SystemUtility();

        void load_depth_data(cv::Mat &depth_image, int frame_index);
        void get_in_range_depth(cv::Mat &depth_image);
        void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat &X, cv::Mat &Y);
        void create_spatial_kernel(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);
        void gaussian_distance_weight(cv::Mat &X, cv::Mat &Y, cv::Mat &weight_d, const float sigma_d);

        struct RayCastingData
        {
            cv::Mat surface_prediction_x;
            cv::Mat surface_prediction_y;
            cv::Mat surface_prediction_z;
            cv::Mat surface_prediction_normal_x;
            cv::Mat surface_prediction_normal_y;
            cv::Mat surface_prediction_normal_z;
            cv::Mat traversal_recording;
        };

        struct DepthImage3dData
        {
            cv::Mat depth_image_next;
            cv::Mat raw_vectirces_x;
            cv::Mat raw_vectirces_y;
            cv::Mat raw_normal_x;
            cv::Mat raw_normal_y;
            cv::Mat raw_normal_z;
            cv::Mat vertex_mask;
            cv::Mat bilateral_output;
        };

        std::unique_ptr<RayCastingData> rayCastingData;
        std::unique_ptr<DepthImage3dData> depthData;

        const int WIDTH = 640;
        const int HEIGHT = 480;

    private:
    };
}

#endif