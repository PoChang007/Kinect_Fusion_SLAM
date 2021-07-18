#include <iostream>
#include "system_utility.h"

namespace Kinfu
{
    SystemUtility::SystemUtility()
    {
        rayCastingData = std::make_unique<RayCastingData>();
        rayCastingData->surface_prediction_x = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        rayCastingData->surface_prediction_y = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        rayCastingData->surface_prediction_z = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        rayCastingData->surface_prediction_normal_x = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        rayCastingData->surface_prediction_normal_y = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        rayCastingData->surface_prediction_normal_z = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        rayCastingData->traversal_recording = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);

        depthData = std::make_unique<DepthImage3dData>();
        depthData->depth_image_next = cv::Mat::zeros(HEIGHT, WIDTH, CV_16UC1);
        depthData->raw_vectirces_x = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        depthData->raw_vectirces_y = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        depthData->raw_normal_x = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        depthData->raw_normal_y = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        depthData->raw_normal_z = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
        depthData->vertex_mask = cv::Mat::zeros(HEIGHT, WIDTH, CV_8U);
        depthData->bilateral_output = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
    }

    SystemUtility::~SystemUtility()
    {
        std::cout << "Data destroy" << std::endl;
        rayCastingData->surface_prediction_x.release();
        rayCastingData->surface_prediction_y.release();
        rayCastingData->surface_prediction_z.release();
        rayCastingData->surface_prediction_normal_x.release();
        rayCastingData->surface_prediction_normal_y.release();
        rayCastingData->surface_prediction_normal_z.release();
        rayCastingData->traversal_recording.release();

        depthData->depth_image_next.release();
        depthData->raw_vectirces_x.release();
        depthData->raw_vectirces_y.release();
        depthData->raw_normal_x.release();
        depthData->raw_normal_y.release();
        depthData->raw_normal_z.release();
        depthData->vertex_mask.release();
        depthData->bilateral_output.release();
    }

    void SystemUtility::load_depth_data(cv::Mat &depth_image, int frame_index)
    {
        char filename[200];
        sprintf(filename, "../data/dep%04d.dat", frame_index);
        FILE *fp = fopen(filename, "rb");
        fread(depth_image.data, 2, HEIGHT * WIDTH, fp);
        fclose(fp);

        // convert to float type
        depth_image.convertTo(depth_image, CV_32F);
    }

    void SystemUtility::get_in_range_depth(cv::Mat &depth_image)
    {
        // filter out uninterested regions
        float MaxRange = 2000.f;
        float MinRange = 300.f;
        cv::Mat FilterImage;
        // Split into 255 and 0
        cv::inRange(depth_image, MinRange, MaxRange, FilterImage); // get either 255 or 0
        FilterImage = FilterImage / 255;
        FilterImage.convertTo(FilterImage, CV_32F);
        depth_image = depth_image.mul(FilterImage);
    }

    void SystemUtility::meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat &X, cv::Mat &Y)
    {
        cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
        cv::repeat(ygv, 1, xgv.total(), Y);
    }

    void SystemUtility::create_spatial_kernel(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
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
        meshgrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
    }

    void SystemUtility::gaussian_distance_weight(cv::Mat &X, cv::Mat &Y, cv::Mat &weight_d, const float sigma_d)
    {
        cv::Mat temp;
        temp = -(X.mul(X, 1) + Y.mul(Y, 1)) / (2 * sigma_d * sigma_d);
        cv::exp(temp, weight_d);
    }
}