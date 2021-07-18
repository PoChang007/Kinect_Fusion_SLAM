#ifndef KINFU_PIPELINE_H_
#define KINFU_PIPELINE_H_

#include <iostream>
#include <thread>
#include "system_utility.h"
#include "parallel_processing.h"

namespace Kinfu
{
    class KinfuPipeline
    {
    public:
        KinfuPipeline();
        ~KinfuPipeline();

        cv::Mat intrinsic_matrix;
        cv::Mat extrinsic_matrix;

        // parameters for kinect fusion environment
        const int start_frame = 20;
        const int next_frame = 21;
        const int per_nth_frame = 1;
        const int end_frame = 70;

        // voxel size, global tsdf storage
        const int voxel_grid_x_start = -80;
        const int voxel_grid_y_start = -80;
        const int voxel_grid_z_start = -30;
        const int voxel_length = 160;
        const int voxel_width = 250;
        const int voxel_height = 140;
        const int ARRAY_SIZE = voxel_length * voxel_width * voxel_height;
        const float voxel_distance = 10;

        float *voxel_grid_x;
        float *voxel_grid_y;
        float *voxel_grid_z;
        float *global_tsdf;
        float *global_weight_tsdf;

        float truncated_distance = 300.f;
        // Maximumand Minimum SDF value
        float sdf_maximum = 300.f;
        float sdf_minimum = -30.f;

        float *depth_Image_index_y;
        float *depth_Image_index_x;

        cv::Mat depth_image;

        void StartProcessing();

    private:
        std::unique_ptr<SystemUtility> _systemUtility;
    };
}

#endif