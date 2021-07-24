#include "kinfu_pipeline.h"

namespace Kinfu
{
    KinfuPipeline::KinfuPipeline(int height, int width)
    {
        _system_utility = std::make_unique<SystemUtility>(height, width);

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
        depth_Image_index_y = new float[width * height];
        depth_Image_index_x = new float[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                depth_Image_index_y[y * width + x] = (float)y;
                depth_Image_index_x[y * width + x] = (float)x;
            }
        }

        // initialize voxel grid
        voxel_grid_x = new float[_array_size];
        voxel_grid_y = new float[_array_size];
        voxel_grid_z = new float[_array_size];

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
        global_tsdf = new float[_array_size];
        global_weight_tsdf = new float[_array_size];
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

        delete[] depth_Image_index_y;
        delete[] depth_Image_index_x;
        delete[] voxel_grid_x;
        delete[] voxel_grid_y;
        delete[] voxel_grid_z;
        delete[] global_tsdf;
        delete[] global_weight_tsdf;
    }

    void KinfuPipeline::InitialProcessing(int start_frame)
    {
        _system_utility->LoadDepthData(_system_utility->initial_depth_image, start_frame);
        _system_utility->GetRangeDepth(_system_utility->initial_depth_image);
        _system_utility->LoadColorData(_system_utility->color_image, start_frame);

        Projective_TSDF(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(),
                        voxel_grid_x, voxel_grid_y, voxel_grid_z, intrinsic_matrix, extrinsic_matrix,
                        global_tsdf, global_weight_tsdf, _system_utility->initial_depth_image,
                        _truncated_distance, _sdf_minimum, _sdf_maximum,
                        _voxel_length, _voxel_width, _voxel_height, true);

        Ray_Casting(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(),
                    _system_utility->ray_casting_data->surface_prediction_x,
                    _system_utility->ray_casting_data->surface_prediction_y,
                    _system_utility->ray_casting_data->surface_prediction_z,
                    _system_utility->ray_casting_data->surface_prediction_normal_x,
                    _system_utility->ray_casting_data->surface_prediction_normal_y,
                    _system_utility->ray_casting_data->surface_prediction_normal_z,
                    voxel_grid_x, voxel_grid_y, voxel_grid_z,
                    depth_Image_index_y, depth_Image_index_x,
                    intrinsic_matrix, extrinsic_matrix, global_tsdf, _system_utility->initial_depth_image,
                    _system_utility->ray_casting_data->traversal_recording, _truncated_distance,
                    _voxel_length, _voxel_width, _voxel_height,
                    _voxel_grid_x_start, _voxel_grid_y_start, _voxel_grid_z_start, _voxel_distance);

        _system_utility->CreateSpatialKernel(cv::Range(-_bw_radius, _bw_radius), cv::Range(-_bw_radius, _bw_radius), _spatial_kernel_y, _spatial_kernel_x);
        _system_utility->GaussianDistanceWeight(_spatial_kernel_y, _spatial_kernel_x, _weight_d, _sigma_d);
    }

    void KinfuPipeline::IncomingFrameProcessing(int current_frame, int per_nth_frame)
    {
        auto start = std::chrono::high_resolution_clock::now();

        _system_utility->depth_data->depth_image_next.convertTo(_system_utility->depth_data->depth_image_next, CV_16UC1);
        CleanDepthData();

        _system_utility->LoadDepthData(_system_utility->depth_data->depth_image_next, current_frame);
        _system_utility->GetRangeDepth(_system_utility->depth_data->depth_image_next);
        _system_utility->LoadColorData(_system_utility->color_image, current_frame);
        Calculate_Vertices_And_Normals(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(),
                                       _system_utility->depth_data->depth_image_next, intrinsic_matrix,
                                       _system_utility->depth_data->raw_vertices_x,
                                       _system_utility->depth_data->raw_vertices_y,
                                       _system_utility->depth_data->raw_normal_x,
                                       _system_utility->depth_data->raw_normal_y,
                                       _system_utility->depth_data->raw_normal_z,
                                       _system_utility->depth_data->vertex_mask,
                                       depth_Image_index_y, depth_Image_index_x);

        // Bilateral_Filtering(_system_utility->depth_data->depth_image_next,
        //                     _system_utility->depth_data->bilateral_output,
        //                     _weight_d, depth_Image_index_y, depth_Image_index_x,
        // 	                _bw_radius, _sigma_r);

        Estimate_Sensor_Pose(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(),
                             intrinsic_matrix, extrinsic_matrix,
                             _system_utility->depth_data->raw_vertices_x,
                             _system_utility->depth_data->raw_vertices_y,
                             _system_utility->depth_data->depth_image_next,
                             _system_utility->depth_data->raw_normal_x,
                             _system_utility->depth_data->raw_normal_y,
                             _system_utility->depth_data->raw_normal_z,
                             _system_utility->ray_casting_data->surface_prediction_x,
                             _system_utility->ray_casting_data->surface_prediction_y,
                             _system_utility->ray_casting_data->surface_prediction_z,
                             _system_utility->ray_casting_data->surface_prediction_normal_x,
                             _system_utility->ray_casting_data->surface_prediction_normal_y,
                             _system_utility->ray_casting_data->surface_prediction_normal_z,
                             _system_utility->depth_data->vertex_mask);

        printf("frame %d\n", current_frame);
        printf("03 %f\n", extrinsic_matrix.ptr<float>(0)[3]);
        printf("13 %f\n", extrinsic_matrix.ptr<float>(1)[3]);
        printf("23 %f\n", extrinsic_matrix.ptr<float>(2)[3]);

        Projective_TSDF(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(),
                        voxel_grid_x, voxel_grid_y, voxel_grid_z, intrinsic_matrix, extrinsic_matrix,
                        global_tsdf, global_weight_tsdf,
                        _system_utility->depth_data->depth_image_next, _truncated_distance, _sdf_minimum, _sdf_maximum,
                        _voxel_length, _voxel_width, _voxel_height, false);

        CleanRayCastingData();

        Ray_Casting(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(),
                    _system_utility->ray_casting_data->surface_prediction_x,
                    _system_utility->ray_casting_data->surface_prediction_y,
                    _system_utility->ray_casting_data->surface_prediction_z,
                    _system_utility->ray_casting_data->surface_prediction_normal_x,
                    _system_utility->ray_casting_data->surface_prediction_normal_y,
                    _system_utility->ray_casting_data->surface_prediction_normal_z,
                    voxel_grid_x, voxel_grid_y, voxel_grid_z,
                    depth_Image_index_y, depth_Image_index_x,
                    intrinsic_matrix, extrinsic_matrix, global_tsdf, _system_utility->depth_data->depth_image_next,
                    _system_utility->ray_casting_data->traversal_recording, _truncated_distance,
                    _voxel_length, _voxel_width, _voxel_height,
                    _voxel_grid_x_start, _voxel_grid_y_start, _voxel_grid_z_start, _voxel_distance);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << current_frame << " frame elapsed time: " << elapsed.count() << " s\n";
        current_frame += per_nth_frame;
    }

    void KinfuPipeline::CleanRayCastingData()
    {
        _system_utility->ray_casting_data->surface_prediction_x = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->ray_casting_data->surface_prediction_y = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->ray_casting_data->surface_prediction_z = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->ray_casting_data->surface_prediction_normal_x = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->ray_casting_data->surface_prediction_normal_y = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->ray_casting_data->surface_prediction_normal_z = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->ray_casting_data->traversal_recording = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
    }

    void KinfuPipeline::CleanDepthData()
    {
        _system_utility->depth_data->depth_image_next = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_16UC1);
        _system_utility->depth_data->raw_vertices_x = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->depth_data->raw_vertices_y = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->depth_data->raw_normal_x = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->depth_data->raw_normal_y = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->depth_data->raw_normal_z = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
        _system_utility->depth_data->vertex_mask = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_8U);
        _system_utility->depth_data->bilateral_output = cv::Mat::zeros(_system_utility->GetImageHeight(), _system_utility->GetImageWidth(), CV_32F);
    }
}