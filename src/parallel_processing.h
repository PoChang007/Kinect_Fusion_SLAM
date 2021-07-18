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

extern "C" void projective_tsdf(float *voxel_grid_x, float *voxel_grid_y, float *voxel_grid_z,
                                cv::Mat &intrinsic_matrix, cv::Mat &extrinsic_matrix, float *global_tsdf, float *global_weight_tsdf,
                                cv::Mat &depth_image, float truncated_distance, float sdf_minimum, float sdf_maximum,
                                int voxel_length, int voxel_width, int voxel_height, bool first_tsdf_construct);

extern "C" void ray_casting(cv::Mat &surface_prediction_x, cv::Mat &surface_prediction_y, cv::Mat &surface_prediction_z,
                            cv::Mat &surface_prediction_normal_x, cv::Mat &surface_prediction_normal_y, cv::Mat &surface_prediction_normal_z,
                            float *voxel_grid_x, float *voxel_grid_y, float *voxel_grid_z,
                            float *depth_Image_index_y, float *depth_Image_index_x,
                            cv::Mat &intrinsic_matrix, cv::Mat &extrinsic_matrix, float *global_tsdf,
                            cv::Mat &depth_image, cv::Mat &traversal_recording, float truncated_distance,
                            int voxel_length, int voxel_width, int voxel_height,
                            int voxel_grid_x_start, int voxel_grid_y_start, int voxel_grid_z_start, float voxel_distance);

extern "C" void calcaulte_vectices_and_normals(cv::Mat &depth_image_next, cv::Mat &intrinsic_matrix,
                                               cv::Mat &raw_vectirces_x, cv::Mat &raw_vectirces_y,
                                               cv::Mat &raw_normal_x, cv::Mat &raw_normal_y, cv::Mat &raw_normal_z,
                                               cv::Mat &vertex_mask, float *depth_Image_index_y, float *depth_Image_index_x);

extern "C" void estimate_sensor_pose(cv::Mat &intrinsic_matrix, cv::Mat &extrinsic_matrix,
                                     cv::Mat &raw_vectirces_x, cv::Mat &raw_vectirces_y, cv::Mat &depth_image_next,
                                     cv::Mat &raw_normal_x, cv::Mat &raw_normal_y, cv::Mat &raw_normal_z,
                                     cv::Mat &surface_prediction_x, cv::Mat &surface_prediction_y, cv::Mat &surface_prediction_z,
                                     cv::Mat &surface_prediction_normal_x, cv::Mat &surface_prediction_normal_y, cv::Mat &surface_prediction_normal_z,
                                     cv::Mat &vertex_mask);

extern "C" void bilateral_filtering(cv::Mat &depth_Image, cv::Mat &bilateral_output,
                                    cv::Mat &weight_d, float *depth_Image_index_y, float *depth_Image_index_x,
                                    const int bw_radius, const float sigma_r);

#endif