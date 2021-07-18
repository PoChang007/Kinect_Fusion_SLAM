#ifndef PARALLEL_PROCESSING_H_
#define PARALLEL_PROCESSING_H_

#include <iostream>

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

extern "C" void Projective_TSDF(const float *voxel_grid_x, const float *voxel_grid_y, const float *voxel_grid_z,
                                const cv::Mat &intrinsic_matrix, cv::Mat &extrinsic_matrix, float *global_tsdf, float *global_weight_tsdf,
                                cv::Mat &depth_image, const float truncated_distance, const float sdf_minimum, const float sdf_maximum,
                                const int voxel_length, const int voxel_width, const int voxel_height, bool first_tsdf_construct);

extern "C" void Ray_Casting(cv::Mat &surface_prediction_x, cv::Mat &surface_prediction_y, cv::Mat &surface_prediction_z,
                            cv::Mat &surface_prediction_normal_x, cv::Mat &surface_prediction_normal_y, cv::Mat &surface_prediction_normal_z,
                            const float *voxel_grid_x, const float *voxel_grid_y, const float *voxel_grid_z,
                            const float *depth_Image_index_y, const float *depth_Image_index_x,
                            const cv::Mat &intrinsic_matrix, cv::Mat &extrinsic_matrix, float *global_tsdf,
                            cv::Mat &depth_image, cv::Mat &traversal_recording, const float truncated_distance,
                            const int voxel_length, const int voxel_width, const int voxel_height,
                            const int voxel_grid_x_start, const int voxel_grid_y_start, const int voxel_grid_z_start, const float voxel_distance);

extern "C" void Calculate_Vertices_And_Normals(cv::Mat &depth_image_next, cv::Mat &intrinsic_matrix,
                                               cv::Mat &raw_vertices_x, cv::Mat &raw_vertices_y,
                                               cv::Mat &raw_normal_x, cv::Mat &raw_normal_y, cv::Mat &raw_normal_z,
                                               cv::Mat &vertex_mask, const float *depth_Image_index_y, const float *depth_Image_index_x);

extern "C" void Estimate_Sensor_Pose(const cv::Mat &intrinsic_matrix, cv::Mat &extrinsic_matrix,
                                     cv::Mat &raw_vertices_x, cv::Mat &raw_vertices_y, cv::Mat &depth_image_next,
                                     cv::Mat &raw_normal_x, cv::Mat &raw_normal_y, cv::Mat &raw_normal_z,
                                     cv::Mat &surface_prediction_x, cv::Mat &surface_prediction_y, cv::Mat &surface_prediction_z,
                                     cv::Mat &surface_prediction_normal_x, cv::Mat &surface_prediction_normal_y, cv::Mat &surface_prediction_normal_z,
                                     cv::Mat &vertex_mask);

extern "C" void Bilateral_Filtering(cv::Mat &depth_Image, cv::Mat &bilateral_output,
                                    cv::Mat &weight_d, const float *depth_Image_index_y, const float *depth_Image_index_x,
                                    const int bw_radius, const float sigma_r);

#endif