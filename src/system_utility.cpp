#include <iostream>
#include "system_utility.h"

namespace Kinfu
{
    SystemUtility::SystemUtility(int height, int width, float max_depth, float min_depth)
    {
        _height = height;
        _width = width;
        _max_depth = max_depth;
        _min_depth = min_depth;
        initial_depth_image = cv::Mat::zeros(_height, _width, CV_16UC1);
        color_image = cv::Mat::zeros(_height, _width, CV_8UC3);

        ray_casting_data = std::make_unique<RayCastingData>();
        ray_casting_data->surface_prediction_x = cv::Mat::zeros(_height, _width, CV_32F);
        ray_casting_data->surface_prediction_y = cv::Mat::zeros(_height, _width, CV_32F);
        ray_casting_data->surface_prediction_z = cv::Mat::zeros(_height, _width, CV_32F);
        ray_casting_data->surface_prediction_normal_x = cv::Mat::zeros(_height, _width, CV_32F);
        ray_casting_data->surface_prediction_normal_y = cv::Mat::zeros(_height, _width, CV_32F);
        ray_casting_data->surface_prediction_normal_z = cv::Mat::zeros(_height, _width, CV_32F);
        ray_casting_data->traversal_recording = cv::Mat::zeros(_height, _width, CV_32F);

        depth_data = std::make_unique<DepthImage3dData>();
        depth_data->depth_image_next = cv::Mat::zeros(_height, _width, CV_16UC1);
        depth_data->raw_vertices_x = cv::Mat::zeros(_height, _width, CV_32F);
        depth_data->raw_vertices_y = cv::Mat::zeros(_height, _width, CV_32F);
        depth_data->raw_normal_x = cv::Mat::zeros(_height, _width, CV_32F);
        depth_data->raw_normal_y = cv::Mat::zeros(_height, _width, CV_32F);
        depth_data->raw_normal_z = cv::Mat::zeros(_height, _width, CV_32F);
        depth_data->vertex_mask = cv::Mat::zeros(_height, _width, CV_8U);
        depth_data->bilateral_output = cv::Mat::zeros(_height, _width, CV_32F);
    }

    SystemUtility::~SystemUtility()
    {
        initial_depth_image.release();
        color_image.release();

        ray_casting_data->surface_prediction_x.release();
        ray_casting_data->surface_prediction_y.release();
        ray_casting_data->surface_prediction_z.release();
        ray_casting_data->surface_prediction_normal_x.release();
        ray_casting_data->surface_prediction_normal_y.release();
        ray_casting_data->surface_prediction_normal_z.release();
        ray_casting_data->traversal_recording.release();

        depth_data->depth_image_next.release();
        depth_data->raw_vertices_x.release();
        depth_data->raw_vertices_y.release();
        depth_data->raw_normal_x.release();
        depth_data->raw_normal_y.release();
        depth_data->raw_normal_z.release();
        depth_data->vertex_mask.release();
        depth_data->bilateral_output.release();
    }

    void SystemUtility::LoadDepthData(cv::Mat &depth_image, int frame_index)
    {
        char filename[200];
        sprintf(filename, "../data/dep%04d.dat", frame_index);
        FILE *fp = fopen(filename, "rb");
        fread(depth_image.data, 2, _height * _width, fp);
        fclose(fp);

        // convert to float type
        depth_image.convertTo(depth_image, CV_32F);
    }

    void SystemUtility::LoadColorData(cv::Mat &color_image, int frame_index)
    {
        char filename[200];
        sprintf(filename, "../data/img%04d.jpg", frame_index);
        color_image = cv::imread(filename);
    }

    void SystemUtility::GetRangeDepth(cv::Mat &depth_image)
    {
        // filter out uninterested regions
        cv::Mat filter_image;
        // Split into 255 and 0
        cv::inRange(depth_image, _min_depth, _max_depth, filter_image); // get either 255 or 0
        filter_image = filter_image / 255;
        filter_image.convertTo(filter_image, CV_32F);
        depth_image = depth_image.mul(filter_image);
    }

    void SystemUtility::GenerateMeshGrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat &X, cv::Mat &Y)
    {
        cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
        cv::repeat(ygv, 1, xgv.total(), Y);
    }

    void SystemUtility::CreateSpatialKernel(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
    {
        std::vector<float> t_x, t_y;
        for (float i = xgv.start; i <= xgv.end; i++)
        {
            t_x.push_back(i);
        }
        for (float i = ygv.start; i <= ygv.end; i++)
        {
            t_y.push_back(i);
        }
        GenerateMeshGrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
    }

    void SystemUtility::GaussianDistanceWeight(cv::Mat &X, cv::Mat &Y, cv::Mat &weight_d, const float sigma_d)
    {
        cv::Mat temp;
        temp = -(X.mul(X, 1) + Y.mul(Y, 1)) / (2 * sigma_d * sigma_d);
        cv::exp(temp, weight_d);
    }
}