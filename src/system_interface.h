#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <numeric>
#include <chrono>

// OpenCV
#include <opencv2/highgui.hpp>  
#include <opencv2/core.hpp>

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <math_constants.h>

#define WIDTH 640
#define HEIGHT 480
#define INTRINSIC_ELEMENTS 9
#define EXTRINSIC_ELEMENTS 16

using namespace std;
using namespace cv;

void initialize_voxel_grid(float* voxel_grid_x, float* voxel_grid_y, float* voxel_grid_z, 
	                       int voxel_grid_x_start, int voxel_grid_y_start, int voxel_grid_z_start,
	                       int voxel_length, int voxel_width, int voxel_height, float voxel_distance);
void initialize_int_and_ext_matrix(Mat& intrinsic_matrix, Mat& extrinsic_matrix);
void load_depth_data(Mat& depth_image, int frame_index);
void get_in_range_depth(Mat& depth_image);
void initialize_index_matrix(float* depth_Image_index_y, float* depth_Image_index_x);
void create_spatial_kernel(const cv::Range& xgv, const cv::Range& ygv,
	                       cv::Mat& X, cv::Mat& Y);
void gaussian_distance_weight(Mat& spatial_kernel_x, Mat& spatial_kernel_y, Mat& weight_d, const float sigma_d);

void  projective_tsdf_cpp(float* voxel_grid_x, float* voxel_grid_y, float* voxel_grid_z,
	                      Mat& intrinsic_matrix, Mat& extrinsic_matrix, float* global_tsdf, float* global_weight_tsdf,
	                      Mat& depth_image, float truncated_distance, float sdf_minimum, float sdf_maximum,
	                      int voxel_length, int voxel_width, int voxel_height, bool first_tsdf_construct);

////////////////////////////////////////////////////////////////////////////////
// functions in cuda files
extern "C" void bilateral_filtering(Mat & depth_Image, Mat & bilateral_output,
                                    Mat & weight_d, float* depth_Image_index_y, float* depth_Image_index_x,
                                    const int bw_radius, const float sigma_r);

extern "C" void projective_tsdf(float* voxel_grid_x, float* voxel_grid_y, float* voxel_grid_z,
                                Mat & intrinsic_matrix, Mat & extrinsic_matrix, float* global_tsdf, float* global_weight_tsdf,
                                Mat & depth_image, float truncated_distance, float sdf_minimum, float sdf_maximum,
                                int voxel_length, int voxel_width, int voxel_height, bool first_tsdf_construct);

extern "C" void ray_casting(Mat & surface_prediction_x, Mat & surface_prediction_y, Mat & surface_prediction_z,
                            Mat & surface_prediction_normal_x, Mat & surface_prediction_normal_y, Mat & surface_prediction_normal_z,
                            float* voxel_grid_x, float* voxel_grid_y, float* voxel_grid_z,
                            float* depth_Image_index_y, float* depth_Image_index_x,
                            Mat & intrinsic_matrix, Mat & extrinsic_matrix, float* global_tsdf,
                            Mat & depth_image, Mat & traversal_recording, float truncated_distance,
                            int voxel_length, int voxel_width, int voxel_height,
                            int voxel_grid_x_start, int voxel_grid_y_start, int voxel_grid_z_start, float voxel_distance);

extern "C" void calcaulte_vectices_and_normals(Mat & depth_image_next, Mat & intrinsic_matrix,
                                               Mat & raw_vectirces_x, Mat & raw_vectirces_y,
                                               Mat & raw_normal_x, Mat & raw_normal_y, Mat & raw_normal_z,
                                               Mat & vertex_mask, float* depth_Image_index_y, float* depth_Image_index_x);

extern "C" void estimate_sensor_pose(Mat & intrinsic_matrix, Mat & extrinsic_matrix,
                                     Mat & raw_vectirces_x, Mat & raw_vectirces_y, Mat & depth_image_next,
                                     Mat & raw_normal_x, Mat & raw_normal_y, Mat & raw_normal_z,
                                     Mat & surface_prediction_x, Mat & surface_prediction_y, Mat & surface_prediction_z,
                                     Mat & surface_prediction_normal_x, Mat & surface_prediction_normal_y, Mat & surface_prediction_normal_z,
                                     Mat & vertex_mask);