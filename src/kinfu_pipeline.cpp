#include "kinfu_pipeline.h"

namespace Kinfu
{
    KinfuPipeline::KinfuPipeline()
    {
        _systemUtility = std::make_unique<SystemUtility>();

        intrinsic_matrix = cv::Mat::zeros(3, 3, CV_32F);
        extrinsic_matrix = cv::Mat::eye(4, 4, CV_32F);
        depth_image = cv::Mat::zeros(HEIGHT, WIDTH, CV_16UC1);

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
        depth_Image_index_y = new float[WIDTH * HEIGHT];
        depth_Image_index_x = new float[WIDTH * HEIGHT];
        for (int y = 0; y < HEIGHT; y++)
        {
            for (int x = 0; x < WIDTH; x++)
            {
                depth_Image_index_y[y * WIDTH + x] = (float)y;
                depth_Image_index_x[y * WIDTH + x] = (float)x;
            }
        }

        // initialize voxel grid
        voxel_grid_x = new float[ARRAY_SIZE];
        voxel_grid_y = new float[ARRAY_SIZE];
        voxel_grid_z = new float[ARRAY_SIZE];

        int x_index_from_zero = -voxel_grid_x_start;
        int y_index_from_zero = -voxel_grid_y_start;
        int z_index_from_zero = -voxel_grid_z_start;

        for (int x = voxel_grid_x_start; x < voxel_grid_x_start + voxel_length; x++)
        {
            for (int y = voxel_grid_y_start; y < voxel_grid_y_start + voxel_width; y++)
            {
                for (int k = voxel_grid_z_start; k < voxel_grid_z_start + voxel_height; k++)
                {
                    voxel_grid_x[(x + x_index_from_zero) * voxel_width * voxel_height + (y + y_index_from_zero) * voxel_height + (k + z_index_from_zero)] = x * voxel_distance + voxel_distance / 2.f;
                    voxel_grid_y[(x + x_index_from_zero) * voxel_width * voxel_height + (y + y_index_from_zero) * voxel_height + (k + z_index_from_zero)] = y * voxel_distance + voxel_distance / 2.f;
                    voxel_grid_z[(x + x_index_from_zero) * voxel_width * voxel_height + (y + y_index_from_zero) * voxel_height + (k + z_index_from_zero)] = k * voxel_distance + voxel_distance / 2.f;
                }
            }
        }

        // for truncated signed distane value storage
        global_tsdf = new float[ARRAY_SIZE];
        global_weight_tsdf = new float[ARRAY_SIZE];
        std::fill_n(global_tsdf, ARRAY_SIZE, NAN);
        std::fill_n(global_weight_tsdf, ARRAY_SIZE, NAN);
    }

    KinfuPipeline::~KinfuPipeline()
    {
        delete[] depth_Image_index_y;
        delete[] depth_Image_index_x;
        delete[] voxel_grid_x;
        delete[] voxel_grid_y;
        delete[] voxel_grid_z;
        delete[] global_tsdf;
        delete[] global_weight_tsdf;
    }

    void KinfuPipeline::StartProcessing()
    {
        _systemUtility->load_depth_data(depth_image, start_frame);
        _systemUtility->get_in_range_depth(depth_image);

        projective_tsdf(voxel_grid_x, voxel_grid_y, voxel_grid_z, intrinsic_matrix, extrinsic_matrix,
                        global_tsdf, global_weight_tsdf,
                        depth_image, truncated_distance, sdf_minimum, sdf_maximum,
                        voxel_length, voxel_width, voxel_height, true);

        ray_casting(_systemUtility->rayCastingData->surface_prediction_x,
                    _systemUtility->rayCastingData->surface_prediction_y,
                    _systemUtility->rayCastingData->surface_prediction_z,
                    _systemUtility->rayCastingData->surface_prediction_normal_x,
                    _systemUtility->rayCastingData->surface_prediction_normal_y,
                    _systemUtility->rayCastingData->surface_prediction_normal_z,
                    voxel_grid_x, voxel_grid_y, voxel_grid_z,
                    depth_Image_index_y, depth_Image_index_x,
                    intrinsic_matrix, extrinsic_matrix, global_tsdf, depth_image,
                    _systemUtility->rayCastingData->traversal_recording, truncated_distance,
                    voxel_length, voxel_width, voxel_height,
                    voxel_grid_x_start, voxel_grid_y_start, voxel_grid_z_start, voxel_distance);

        // spatial Kernel
        cv::Mat spatial_kernel_y;
        cv::Mat spatial_kernel_x;

        // filter window
        const int bw_radius = 3;
        _systemUtility->create_spatial_kernel(cv::Range(-bw_radius, bw_radius), cv::Range(-bw_radius, bw_radius), spatial_kernel_y, spatial_kernel_x);

        // computing Gaussian distance weight
        const float sigma_d = 4.5f;
        const float sigma_r = 30.f;
        cv::Mat weight_d;
        _systemUtility->gaussian_distance_weight(spatial_kernel_y, spatial_kernel_x, weight_d, sigma_d);

        // loop process
        // calculate vertices and normals in the next frame
        for (int i = next_frame; i <= end_frame; i += per_nth_frame)
        {
            auto start = std::chrono::high_resolution_clock::now();

            _systemUtility->depthData->depth_image_next.convertTo(_systemUtility->depthData->depth_image_next, CV_16UC1);
            _systemUtility->depthData->depth_image_next = cv::Mat::zeros(HEIGHT, WIDTH, CV_16UC1);
            _systemUtility->depthData->raw_vectirces_x = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->depthData->raw_vectirces_y = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->depthData->raw_normal_x = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->depthData->raw_normal_y = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->depthData->raw_normal_z = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->depthData->vertex_mask = cv::Mat::zeros(HEIGHT, WIDTH, CV_8U);
            _systemUtility->depthData->bilateral_output = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);

            _systemUtility->load_depth_data(_systemUtility->depthData->depth_image_next, i);
            _systemUtility->get_in_range_depth(_systemUtility->depthData->depth_image_next);
            calcaulte_vectices_and_normals(_systemUtility->depthData->depth_image_next, intrinsic_matrix,
                                           _systemUtility->depthData->raw_vectirces_x,
                                           _systemUtility->depthData->raw_vectirces_y,
                                           _systemUtility->depthData->raw_normal_x,
                                           _systemUtility->depthData->raw_normal_y,
                                           _systemUtility->depthData->raw_normal_z,
                                           _systemUtility->depthData->vertex_mask,
                                           depth_Image_index_y, depth_Image_index_x);

            // bilateral_filtering(depth_image_next, bilateral_output, weight_d,
            // 	                depth_Image_index_y, depth_Image_index_x,
            // 	                bw_radius, sigma_r);

            estimate_sensor_pose(intrinsic_matrix, extrinsic_matrix,
                                 _systemUtility->depthData->raw_vectirces_x,
                                 _systemUtility->depthData->raw_vectirces_y,
                                 _systemUtility->depthData->depth_image_next,
                                 _systemUtility->depthData->raw_normal_x,
                                 _systemUtility->depthData->raw_normal_y,
                                 _systemUtility->depthData->raw_normal_z,
                                 _systemUtility->rayCastingData->surface_prediction_x,
                                 _systemUtility->rayCastingData->surface_prediction_y,
                                 _systemUtility->rayCastingData->surface_prediction_z,
                                 _systemUtility->rayCastingData->surface_prediction_normal_x,
                                 _systemUtility->rayCastingData->surface_prediction_normal_y,
                                 _systemUtility->rayCastingData->surface_prediction_normal_z,
                                 _systemUtility->depthData->vertex_mask);

            printf("frame %d\n", i);
            printf("03 %f\n", extrinsic_matrix.ptr<float>(0)[3]);
            printf("13 %f\n", extrinsic_matrix.ptr<float>(1)[3]);
            printf("23 %f\n", extrinsic_matrix.ptr<float>(2)[3]);

            projective_tsdf(voxel_grid_x, voxel_grid_y, voxel_grid_z, intrinsic_matrix, extrinsic_matrix,
                            global_tsdf, global_weight_tsdf,
                            _systemUtility->depthData->depth_image_next, truncated_distance, sdf_minimum, sdf_maximum,
                            voxel_length, voxel_width, voxel_height, false);

            _systemUtility->rayCastingData->surface_prediction_x = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->rayCastingData->surface_prediction_y = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->rayCastingData->surface_prediction_z = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->rayCastingData->surface_prediction_normal_x = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->rayCastingData->surface_prediction_normal_y = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->rayCastingData->surface_prediction_normal_z = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);
            _systemUtility->rayCastingData->traversal_recording = cv::Mat::zeros(HEIGHT, WIDTH, CV_32F);

            ray_casting(_systemUtility->rayCastingData->surface_prediction_x,
                        _systemUtility->rayCastingData->surface_prediction_y,
                        _systemUtility->rayCastingData->surface_prediction_z,
                        _systemUtility->rayCastingData->surface_prediction_normal_x,
                        _systemUtility->rayCastingData->surface_prediction_normal_y,
                        _systemUtility->rayCastingData->surface_prediction_normal_z,
                        voxel_grid_x, voxel_grid_y, voxel_grid_z,
                        depth_Image_index_y, depth_Image_index_x,
                        intrinsic_matrix, extrinsic_matrix, global_tsdf, _systemUtility->depthData->depth_image_next,
                        _systemUtility->rayCastingData->traversal_recording, truncated_distance,
                        voxel_length, voxel_width, voxel_height,
                        voxel_grid_x_start, voxel_grid_y_start, voxel_grid_z_start, voxel_distance);

            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << i << " frame elapsed time: " << elapsed.count() << " s\n";
        }
    }
}