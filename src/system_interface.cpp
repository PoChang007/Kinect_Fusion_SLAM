#include "system_interface.h"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
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
    float* voxel_grid_x = new float[ARRAY_SIZE];
    float* voxel_grid_y = new float[ARRAY_SIZE];
    float* voxel_grid_z = new float[ARRAY_SIZE];
    float* global_tsdf = new float[ARRAY_SIZE];
    float* global_weight_tsdf = new float[ARRAY_SIZE];
    std::fill_n(global_tsdf, ARRAY_SIZE, NAN);
    std::fill_n(global_weight_tsdf, ARRAY_SIZE, NAN);
    
    initialize_voxel_grid(voxel_grid_x, voxel_grid_y, voxel_grid_z, 
                          voxel_grid_x_start, voxel_grid_y_start, voxel_grid_z_start,
                          voxel_length, voxel_width, voxel_height, voxel_distance);

    cv::Mat intrinsic_matrix(3, 3, CV_32F);
    cv::Mat extrinsic_matrix(4, 4, CV_32F);
    initialize_int_and_ext_matrix(intrinsic_matrix, extrinsic_matrix);

    float truncated_distance = 300.f;
    // Maximumand Minimum SDF value
    float sdf_maximum = 300.f;
    float sdf_minimum = -30.f;

    cv::Mat depth_image(HEIGHT, WIDTH, CV_16UC1);
    load_depth_data(depth_image, start_frame);
    get_in_range_depth(depth_image);

    projective_tsdf(voxel_grid_x, voxel_grid_y, voxel_grid_z, intrinsic_matrix, extrinsic_matrix, 
                    global_tsdf, global_weight_tsdf,
                    depth_image, truncated_distance, sdf_minimum, sdf_maximum, 
                    voxel_length, voxel_width, voxel_height, true);

    float* depth_Image_index_y = new float[WIDTH * HEIGHT];
    float* depth_Image_index_x = new float[WIDTH * HEIGHT];

    initialize_index_matrix(depth_Image_index_y, depth_Image_index_x);

    cv::Mat surface_prediction_x = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat surface_prediction_y = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat surface_prediction_z = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat surface_prediction_normal_x = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat surface_prediction_normal_y = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat surface_prediction_normal_z = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat traversal_recording = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);

    ray_casting(surface_prediction_x, surface_prediction_y, surface_prediction_z, 
                surface_prediction_normal_x, surface_prediction_normal_y, surface_prediction_normal_z, 
                voxel_grid_x, voxel_grid_y, voxel_grid_z,
                depth_Image_index_y, depth_Image_index_x, 
                intrinsic_matrix, extrinsic_matrix, global_tsdf,
                depth_image, traversal_recording, truncated_distance, voxel_length, voxel_width, voxel_height,
                voxel_grid_x_start, voxel_grid_y_start, voxel_grid_z_start, voxel_distance);

    cv::Mat depth_image_next(HEIGHT, WIDTH, CV_16UC1);
    cv::Mat raw_vectirces_x = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat raw_vectirces_y = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat raw_normal_x = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat raw_normal_y = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat raw_normal_z = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat vertex_mask = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_8U);
    cv::Mat raw_vectirces_x_check = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat raw_vectirces_y_check = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::Mat bilateral_output = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);

	// spatial Kernel
	cv::Mat spatial_kernel_y;
	cv::Mat spatial_kernel_x;
	// filter window
	const int bw_radius = 3;
	create_spatial_kernel(cv::Range(-bw_radius, bw_radius), cv::Range(-bw_radius, bw_radius), spatial_kernel_y, spatial_kernel_x);
	//std::cerr << spatial_kernel_x << std::endl;
	//std::cerr << spatial_kernel_y << std::endl;

	// computing Gaussian distance weight
	const float sigma_d = 4.5f;
	const float sigma_r = 30.f;
	cv::Mat weight_d;
	gaussian_distance_weight(spatial_kernel_y, spatial_kernel_x, weight_d, sigma_d);

    // loop process
    // Calculate vertices and normals in the next frame
    for (int i = next_frame; i <= end_frame; i += per_nth_frame)
    {
        auto start = std::chrono::high_resolution_clock::now();

        depth_image_next.convertTo(depth_image_next, CV_16UC1);
        depth_image_next = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_16UC1);
        raw_vectirces_x = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        raw_vectirces_y = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        raw_normal_x = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        raw_normal_y = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        raw_normal_z = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        vertex_mask = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_8U);
        bilateral_output = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);

        load_depth_data(depth_image_next, i);
        get_in_range_depth(depth_image_next);
        calcaulte_vectices_and_normals(depth_image_next, intrinsic_matrix,
                                       raw_vectirces_x, raw_vectirces_y,
                                       raw_normal_x, raw_normal_y, raw_normal_z, vertex_mask,
                                       depth_Image_index_y, depth_Image_index_x);

		//bilateral_filtering(depth_image_next, bilateral_output, weight_d,
		//	                depth_Image_index_y, depth_Image_index_x,
		//	                bw_radius, sigma_r);

        estimate_sensor_pose(intrinsic_matrix, extrinsic_matrix,
                             raw_vectirces_x, raw_vectirces_y, depth_image_next,
                             raw_normal_x, raw_normal_y, raw_normal_z,
                             surface_prediction_x, surface_prediction_y, surface_prediction_z,
                             surface_prediction_normal_x, surface_prediction_normal_y, surface_prediction_normal_z,
                             vertex_mask);
        
        printf("frame %d\n", i);
        printf("03 %f\n", extrinsic_matrix.ptr<float>(0)[3]);
        printf("13 %f\n", extrinsic_matrix.ptr<float>(1)[3]);
        printf("23 %f\n", extrinsic_matrix.ptr<float>(2)[3]);

        projective_tsdf(voxel_grid_x, voxel_grid_y, voxel_grid_z, intrinsic_matrix, extrinsic_matrix,
                        global_tsdf, global_weight_tsdf,
                        depth_image_next, truncated_distance, sdf_minimum, sdf_maximum,
                        voxel_length, voxel_width, voxel_height, false);

        surface_prediction_x = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        surface_prediction_y = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        surface_prediction_z = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        surface_prediction_normal_x = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        surface_prediction_normal_y = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        surface_prediction_normal_z = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);
        traversal_recording = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_32F);

        ray_casting(surface_prediction_x, surface_prediction_y, surface_prediction_z,
                    surface_prediction_normal_x, surface_prediction_normal_y, surface_prediction_normal_z,
                    voxel_grid_x, voxel_grid_y, voxel_grid_z,
                    depth_Image_index_y, depth_Image_index_x,
                    intrinsic_matrix, extrinsic_matrix, global_tsdf,
                    depth_image_next, traversal_recording, truncated_distance, voxel_length, voxel_width, voxel_height,
                    voxel_grid_x_start, voxel_grid_y_start, voxel_grid_z_start, voxel_distance);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << i << " frame elapsed time: " << elapsed.count() << " s\n";
    }

	depth_image_next.release();
	raw_vectirces_x.release();
	raw_vectirces_y.release();
	raw_normal_x.release();
	raw_normal_y.release();
	raw_normal_z.release();
	vertex_mask.release();

	surface_prediction_x.release();
	surface_prediction_y.release();
	surface_prediction_z.release();
	surface_prediction_normal_x.release();
	surface_prediction_normal_y.release();
	surface_prediction_normal_z.release();
	traversal_recording.release();

    delete[] depth_Image_index_y;
    delete[] depth_Image_index_x;

    delete[] voxel_grid_x;
    delete[] voxel_grid_y;
    delete[] voxel_grid_z;
    delete[] global_tsdf;
    delete[] global_weight_tsdf;

    waitKey(50000);
    return 0;
}
