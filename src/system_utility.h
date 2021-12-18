#ifndef SYSTEM_UTILITY_H_
#define SYSTEM_UTILITY_H_

#include <thread>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Kinfu
{
    class KinfuPipeline;

    class SystemUtility
    {
    public:
        SystemUtility(int height, int width, float max_depth, float min_depth);
        ~SystemUtility();

        void LoadDepthData(cv::Mat &depth_image, int frame_index);
        void GetRangeDepth(cv::Mat &depth_image);
        void LoadColorData(cv::Mat &color_image, int frame_index);
        void GenerateMeshGrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat &X, cv::Mat &Y);
        void CreateSpatialKernel(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);
        void GaussianDistanceWeight(cv::Mat &X, cv::Mat &Y, cv::Mat &weight_d, const float sigma_d);

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
            cv::Mat raw_vertices_x;
            cv::Mat raw_vertices_y;
            cv::Mat raw_normal_x;
            cv::Mat raw_normal_y;
            cv::Mat raw_normal_z;
            cv::Mat vertex_mask;
            cv::Mat bilateral_output;
        };

        cv::Mat initial_depth_image;
        cv::Mat color_image;
        std::unique_ptr<RayCastingData> ray_casting_data;
        std::unique_ptr<DepthImage3dData> depth_data;

        int GetImageHeight() const
        {
            return _height;
        }

        int GetImageWidth() const
        {
            return _width;
        }

    private:
        int _height{0};
        int _width{0};
        float _max_depth{0.f};
        float _min_depth{0.f};
    };
}

#endif