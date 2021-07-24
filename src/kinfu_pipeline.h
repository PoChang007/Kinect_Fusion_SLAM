#ifndef KINFU_PIPELINE_H_
#define KINFU_PIPELINE_H_

#include <iostream>
#include <thread>
#include "system_utility.h"
#include "parallel_processing.h"

namespace Kinfu
{
    class KinfuPipeline : public std::enable_shared_from_this<Kinfu::KinfuPipeline>
    {
    public:
        KinfuPipeline(int height, int width);
        ~KinfuPipeline();

        cv::Mat intrinsic_matrix;
        cv::Mat extrinsic_matrix;

        // voxel grid, global tsdf storage
        float *voxel_grid_x;
        float *voxel_grid_y;
        float *voxel_grid_z;
        float *global_tsdf;
        float *global_weight_tsdf;

        // iamge index matrices
        float *depth_Image_index_y;
        float *depth_Image_index_x;

        void InitialProcessing(int start_frame);
        void IncomingFrameProcessing(int current_frame, int per_nth_frame);

        std::shared_ptr<KinfuPipeline> get_shared_this() { return shared_from_this(); }
        std::unique_ptr<SystemUtility> _system_utility;

    private:
        void CleanDepthData();
        void CleanRayCastingData();

        // define voxel size
        const int _voxel_grid_x_start{-80};
        const int _voxel_grid_y_start{-20};
        const int _voxel_grid_z_start{-30};
        const int _voxel_length{160};
        const int _voxel_width{250};
        const int _voxel_height{140};
        const int _array_size = _voxel_length * _voxel_width * _voxel_height;
        const float _voxel_distance{10};

        const float _truncated_distance{300.f};

        // Maximumand Minimum SDF value
        const float _sdf_maximum{300.f};
        const float _sdf_minimum{-30.f};

        // filter window
        const int _bw_radius{3};

        // computing Gaussian distance weight
        const float _sigma_d{4.5f};
        const float _sigma_r{30.f};

        // spatial Kernel
        cv::Mat _spatial_kernel_y;
        cv::Mat _spatial_kernel_x;
        cv::Mat _weight_d;
    };
}

#endif