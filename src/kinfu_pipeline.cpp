#include "parallel_processing/parallel_processing.h"
#include "kinfu_pipeline.h"

namespace Kinfu
{
    KinfuPipeline::KinfuPipeline(int height, int width, float max_depth, float min_depth)
    {
        system_utility = std::make_unique<SystemUtility>(height, width, max_depth, min_depth);

        intrinsic_matrix = cv::Mat::zeros(3, 3, CV_32F);
        extrinsic_matrix = cv::Mat::eye(4, 4, CV_32F);

        // assign intrinsic parameters
        intrinsic_matrix.ptr<float>(0)[0] = 524.639f;
        intrinsic_matrix.ptr<float>(0)[1] = 0.0f;
        intrinsic_matrix.ptr<float>(0)[2] = 316.625f;

        intrinsic_matrix.ptr<float>(1)[0] = 0.0f;
        intrinsic_matrix.ptr<float>(1)[1] = 523.503f;
        intrinsic_matrix.ptr<float>(1)[2] = 256.2318f;

        intrinsic_matrix.ptr<float>(2)[0] = 0.0f;
        intrinsic_matrix.ptr<float>(2)[1] = 0.0f;
        intrinsic_matrix.ptr<float>(2)[2] = 1.0f;

        // initialize image index matrices
        depth_Image_index_y = (float *)malloc(sizeof(float) * width * height);
        depth_Image_index_x = (float *)malloc(sizeof(float) * width * height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                depth_Image_index_y[y * width + x] = (float)y;
                depth_Image_index_x[y * width + x] = (float)x;
            }
        }

        // initialize voxel grid
        voxel_grid_x = (float *)malloc(sizeof(float) * _array_size);
        voxel_grid_y = (float *)malloc(sizeof(float) * _array_size);
        voxel_grid_z = (float *)malloc(sizeof(float) * _array_size);

        int x_index_from_zero = -_voxel_grid_x_start;
        int y_index_from_zero = -_voxel_grid_y_start;
        int z_index_from_zero = -_voxel_grid_z_start;

        for (int x = _voxel_grid_x_start; x < _voxel_grid_x_start + _voxel_length; x++)
        {
            for (int y = _voxel_grid_y_start; y < _voxel_grid_y_start + _voxel_width; y++)
            {
                for (int k = _voxel_grid_z_start; k < _voxel_grid_z_start + _voxel_height; k++)
                {
                    voxel_grid_x[(x + x_index_from_zero) * _voxel_width * _voxel_height + (y + y_index_from_zero) * _voxel_height + (k + z_index_from_zero)] = x * _voxel_distance + _voxel_distance / 2.f;
                    voxel_grid_y[(x + x_index_from_zero) * _voxel_width * _voxel_height + (y + y_index_from_zero) * _voxel_height + (k + z_index_from_zero)] = y * _voxel_distance + _voxel_distance / 2.f;
                    voxel_grid_z[(x + x_index_from_zero) * _voxel_width * _voxel_height + (y + y_index_from_zero) * _voxel_height + (k + z_index_from_zero)] = k * _voxel_distance + _voxel_distance / 2.f;
                }
            }
        }

        // for truncated signed distane value storage
        global_tsdf = (float *)malloc(sizeof(float) * _array_size);
        global_weight_tsdf = (float *)malloc(sizeof(float) * _array_size);
        std::fill_n(global_tsdf, _array_size, NAN);
        std::fill_n(global_weight_tsdf, _array_size, NAN);
    }

    KinfuPipeline::~KinfuPipeline()
    {
        intrinsic_matrix.release();
        extrinsic_matrix.release();
        _spatial_kernel_y.release();
        _spatial_kernel_x.release();
        _weight_d.release();

        if (depth_Image_index_y != nullptr)
            free(depth_Image_index_y);
        if (depth_Image_index_x != nullptr)
            free(depth_Image_index_x);
        if (voxel_grid_x != nullptr)
            free(voxel_grid_x);
        if (voxel_grid_y != nullptr)
            free(voxel_grid_y);
        if (voxel_grid_z != nullptr)
            free(voxel_grid_z);
        if (global_tsdf != nullptr)
            free(global_tsdf);
        if (global_weight_tsdf != nullptr)
            free(global_weight_tsdf);
    }

    void KinfuPipeline::InitialProcessing(int start_frame)
    {
        system_utility->LoadDepthData(system_utility->initial_depth_image, start_frame);
        system_utility->GetRangeDepth(system_utility->initial_depth_image);
        system_utility->LoadColorData(system_utility->color_image, start_frame);

        Projective_TSDF(system_utility->GetImageHeight(), system_utility->GetImageWidth(),
                        voxel_grid_x, voxel_grid_y, voxel_grid_z, intrinsic_matrix, extrinsic_matrix,
                        global_tsdf, global_weight_tsdf, system_utility->initial_depth_image,
                        _truncated_distance, _sdf_minimum, _sdf_maximum,
                        _voxel_length, _voxel_width, _voxel_height, true);

        Ray_Casting(system_utility->GetImageHeight(), system_utility->GetImageWidth(),
                    system_utility->ray_casting_data->surface_prediction_x,
                    system_utility->ray_casting_data->surface_prediction_y,
                    system_utility->ray_casting_data->surface_prediction_z,
                    system_utility->ray_casting_data->surface_prediction_normal_x,
                    system_utility->ray_casting_data->surface_prediction_normal_y,
                    system_utility->ray_casting_data->surface_prediction_normal_z,
                    voxel_grid_x, voxel_grid_y, voxel_grid_z,
                    depth_Image_index_y, depth_Image_index_x,
                    intrinsic_matrix, extrinsic_matrix, global_tsdf, system_utility->initial_depth_image,
                    system_utility->ray_casting_data->traversal_recording, _truncated_distance,
                    _voxel_length, _voxel_width, _voxel_height,
                    _voxel_grid_x_start, _voxel_grid_y_start, _voxel_grid_z_start, _voxel_distance);

        system_utility->CreateSpatialKernel(cv::Range(-_bw_radius, _bw_radius), cv::Range(-_bw_radius, _bw_radius), _spatial_kernel_y, _spatial_kernel_x);
        system_utility->GaussianDistanceWeight(_spatial_kernel_y, _spatial_kernel_x, _weight_d, _sigma_d);
    }

    void KinfuPipeline::IncomingFrameProcessing(int current_frame, int per_nth_frame)
    {
        auto start = std::chrono::high_resolution_clock::now();

        system_utility->depth_data->depth_image_next.convertTo(system_utility->depth_data->depth_image_next, CV_16UC1);
        CleanDepthData();

        system_utility->LoadDepthData(system_utility->depth_data->depth_image_next, current_frame);
        system_utility->GetRangeDepth(system_utility->depth_data->depth_image_next);
        system_utility->LoadColorData(system_utility->color_image, current_frame);
        Calculate_Vertices_And_Normals(system_utility->GetImageHeight(), system_utility->GetImageWidth(),
                                       system_utility->depth_data->depth_image_next, intrinsic_matrix,
                                       system_utility->depth_data->raw_vertices_x,
                                       system_utility->depth_data->raw_vertices_y,
                                       system_utility->depth_data->raw_normal_x,
                                       system_utility->depth_data->raw_normal_y,
                                       system_utility->depth_data->raw_normal_z,
                                       system_utility->depth_data->vertex_mask,
                                       depth_Image_index_y, depth_Image_index_x);

        // Bilateral_Filtering(system_utility->GetImageHeight(), system_utility->GetImageWidth(),
        //                     system_utility->depth_data->depth_image_next,
        //                     system_utility->depth_data->bilateral_output,
        //                     _weight_d, depth_Image_index_y, depth_Image_index_x,
        //                     _bw_radius, _sigma_r);

        Estimate_Sensor_Pose(system_utility->GetImageHeight(), system_utility->GetImageWidth(),
                             intrinsic_matrix, extrinsic_matrix,
                             system_utility->depth_data->raw_vertices_x,
                             system_utility->depth_data->raw_vertices_y,
                             system_utility->depth_data->depth_image_next,
                             system_utility->depth_data->raw_normal_x,
                             system_utility->depth_data->raw_normal_y,
                             system_utility->depth_data->raw_normal_z,
                             system_utility->ray_casting_data->surface_prediction_x,
                             system_utility->ray_casting_data->surface_prediction_y,
                             system_utility->ray_casting_data->surface_prediction_z,
                             system_utility->ray_casting_data->surface_prediction_normal_x,
                             system_utility->ray_casting_data->surface_prediction_normal_y,
                             system_utility->ray_casting_data->surface_prediction_normal_z,
                             system_utility->depth_data->vertex_mask);

        std::cout << "frame " << current_frame << std::endl;
        std::cout << "cam_pos_x " << extrinsic_matrix.ptr<float>(0)[3] << std::endl;
        std::cout << "cam_pos_y " << extrinsic_matrix.ptr<float>(1)[3] << std::endl;
        std::cout << "cam_pos_z " << extrinsic_matrix.ptr<float>(2)[3] << std::endl;

        Projective_TSDF(system_utility->GetImageHeight(), system_utility->GetImageWidth(),
                        voxel_grid_x, voxel_grid_y, voxel_grid_z, intrinsic_matrix, extrinsic_matrix,
                        global_tsdf, global_weight_tsdf,
                        system_utility->depth_data->depth_image_next, _truncated_distance, _sdf_minimum, _sdf_maximum,
                        _voxel_length, _voxel_width, _voxel_height, false);

        CleanRayCastingData();

        Ray_Casting(system_utility->GetImageHeight(), system_utility->GetImageWidth(),
                    system_utility->ray_casting_data->surface_prediction_x,
                    system_utility->ray_casting_data->surface_prediction_y,
                    system_utility->ray_casting_data->surface_prediction_z,
                    system_utility->ray_casting_data->surface_prediction_normal_x,
                    system_utility->ray_casting_data->surface_prediction_normal_y,
                    system_utility->ray_casting_data->surface_prediction_normal_z,
                    voxel_grid_x, voxel_grid_y, voxel_grid_z,
                    depth_Image_index_y, depth_Image_index_x,
                    intrinsic_matrix, extrinsic_matrix, global_tsdf, system_utility->depth_data->depth_image_next,
                    system_utility->ray_casting_data->traversal_recording, _truncated_distance,
                    _voxel_length, _voxel_width, _voxel_height,
                    _voxel_grid_x_start, _voxel_grid_y_start, _voxel_grid_z_start, _voxel_distance);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "elapsed time: " << elapsed.count() << " s\n";
    }

    void KinfuPipeline::CleanRayCastingData()
    {
        system_utility->ray_casting_data->surface_prediction_x = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->ray_casting_data->surface_prediction_y = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->ray_casting_data->surface_prediction_z = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->ray_casting_data->surface_prediction_normal_x = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->ray_casting_data->surface_prediction_normal_y = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->ray_casting_data->surface_prediction_normal_z = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->ray_casting_data->traversal_recording = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
    }

    void KinfuPipeline::CleanDepthData()
    {
        system_utility->depth_data->depth_image_next = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_16UC1);
        system_utility->depth_data->raw_vertices_x = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->depth_data->raw_vertices_y = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->depth_data->raw_normal_x = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->depth_data->raw_normal_y = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->depth_data->raw_normal_z = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
        system_utility->depth_data->vertex_mask = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_8U);
        system_utility->depth_data->bilateral_output = cv::Mat::zeros(system_utility->GetImageHeight(), system_utility->GetImageWidth(), CV_32F);
    }
}