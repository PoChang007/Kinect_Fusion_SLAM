#include "parallel_processing.h"

#define WRK 1.0f

__global__ void Projective_TSDF(const float *dev_cam_intrinsic, float *dev_global_extrinsic,
                                const float *dev_inv_cam_intrinsic, float *dev_inv_global_extrinsic, const float *dev_vertices_z,
                                const float *dev_voxel_grid_x, const float *dev_voxel_grid_y, const float *dev_voxel_grid_z,
                                float *dev_global_tsdf, float *dev_global_weight_tsdf,
                                const float truncated_distance, const float sdf_minimum, const float sdf_maximum, bool initial_tsdf_construct,
                                const float voxel_length, const float voxel_width, const float voxel_height,
                                const int HEIGHT, const int WIDTH)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= voxel_length * voxel_width * voxel_height)
        return;

    // each voxel's mid position
    float3 voxel_mid = make_float3(dev_voxel_grid_x[x], dev_voxel_grid_y[x], dev_voxel_grid_z[x]);
    float local_coordinate_x = dev_inv_global_extrinsic[0] * voxel_mid.x + dev_inv_global_extrinsic[1] * voxel_mid.y + dev_inv_global_extrinsic[2] * voxel_mid.z + dev_inv_global_extrinsic[3];
    float local_coordinate_y = dev_inv_global_extrinsic[4] * voxel_mid.x + dev_inv_global_extrinsic[5] * voxel_mid.y + dev_inv_global_extrinsic[6] * voxel_mid.z + dev_inv_global_extrinsic[7];
    float local_coordinate_z = dev_inv_global_extrinsic[8] * voxel_mid.x + dev_inv_global_extrinsic[9] * voxel_mid.y + dev_inv_global_extrinsic[10] * voxel_mid.z + dev_inv_global_extrinsic[11];

    float local_camera_coordinate_x = dev_cam_intrinsic[0] * local_coordinate_x + dev_cam_intrinsic[1] * local_coordinate_y + dev_cam_intrinsic[2] * local_coordinate_z;
    float local_camera_coordinate_y = dev_cam_intrinsic[3] * local_coordinate_x + dev_cam_intrinsic[4] * local_coordinate_y + dev_cam_intrinsic[5] * local_coordinate_z;
    float local_camera_coordinate_z = dev_cam_intrinsic[6] * local_coordinate_x + dev_cam_intrinsic[7] * local_coordinate_y + dev_cam_intrinsic[8] * local_coordinate_z;

    // search the nearest neighbor pixel location
    int local_projected_pixel_u = (int)roundf(local_camera_coordinate_x / local_camera_coordinate_z);
    int local_projected_pixel_v = (int)roundf(local_camera_coordinate_y / local_camera_coordinate_z);

    // check if 3D voxel's projeted point is in viewing frustum
    if (local_projected_pixel_u > 0 && local_projected_pixel_u <= WIDTH)
    {
        if (local_projected_pixel_v > 0 && local_projected_pixel_v <= HEIGHT)
        {
            if (dev_vertices_z[(local_projected_pixel_v - 1) * WIDTH + (local_projected_pixel_u - 1)] != 0)
            {
                // calcaulte the point in camera coordinate with depth = 1
                float lambda_x = dev_inv_cam_intrinsic[0] * local_projected_pixel_u + dev_inv_cam_intrinsic[1] * local_projected_pixel_v + dev_inv_cam_intrinsic[2] * 1.f;
                float lambda_y = dev_inv_cam_intrinsic[3] * local_projected_pixel_u + dev_inv_cam_intrinsic[4] * local_projected_pixel_v + dev_inv_cam_intrinsic[5] * 1.f;
                float lambda_z = dev_inv_cam_intrinsic[6] * local_projected_pixel_u + dev_inv_cam_intrinsic[7] * local_projected_pixel_v + dev_inv_cam_intrinsic[8] * 1.f;

                float3 ray_start_point = make_float3(dev_global_extrinsic[0] * lambda_x + dev_global_extrinsic[1] * lambda_y + dev_global_extrinsic[2] * lambda_z + dev_global_extrinsic[3],
                                                     dev_global_extrinsic[4] * lambda_x + dev_global_extrinsic[5] * lambda_y + dev_global_extrinsic[6] * lambda_z + dev_global_extrinsic[7],
                                                     dev_global_extrinsic[8] * lambda_x + dev_global_extrinsic[9] * lambda_y + dev_global_extrinsic[10] * lambda_z + dev_global_extrinsic[11]);

                float3 ray_delta_step_point = make_float3(dev_global_extrinsic[0] * 2 * lambda_x + dev_global_extrinsic[1] * 2 * lambda_y + dev_global_extrinsic[2] * 2 * lambda_z + dev_global_extrinsic[3],
                                                          dev_global_extrinsic[4] * 2 * lambda_x + dev_global_extrinsic[5] * 2 * lambda_y + dev_global_extrinsic[6] * 2 * lambda_z + dev_global_extrinsic[7],
                                                          dev_global_extrinsic[8] * 2 * lambda_x + dev_global_extrinsic[9] * 2 * lambda_y + dev_global_extrinsic[10] * 2 * lambda_z + dev_global_extrinsic[11]);

                float3 ray_direction = make_float3(ray_delta_step_point.x - ray_start_point.x, ray_delta_step_point.y - ray_start_point.y, ray_delta_step_point.z - ray_start_point.z);
                float ray_direction_magnitude = sqrtf(ray_direction.x * ray_direction.x + ray_direction.y * ray_direction.y + ray_direction.z * ray_direction.z);
                float3 translation_diff = make_float3(voxel_mid.x - dev_global_extrinsic[3], voxel_mid.y - dev_global_extrinsic[7], voxel_mid.z - dev_global_extrinsic[11]);

                // check to make sure voxel is in front of the camera
                // calcaulte the point along the ray from the central pixel
                float3 camera_coord_point_a = make_float3(dev_inv_cam_intrinsic[0] * dev_cam_intrinsic[5] + dev_inv_cam_intrinsic[1] * dev_cam_intrinsic[2] + dev_inv_cam_intrinsic[2] * 1.f,
                                                          dev_inv_cam_intrinsic[3] * dev_cam_intrinsic[5] + dev_inv_cam_intrinsic[4] * dev_cam_intrinsic[2] + dev_inv_cam_intrinsic[5] * 1.f,
                                                          dev_inv_cam_intrinsic[6] * dev_cam_intrinsic[5] + dev_inv_cam_intrinsic[7] * dev_cam_intrinsic[2] + dev_inv_cam_intrinsic[8] * 1.f);
                float3 global_point_a = make_float3(dev_global_extrinsic[0] * camera_coord_point_a.x + dev_global_extrinsic[1] * camera_coord_point_a.y + dev_global_extrinsic[2] * camera_coord_point_a.z + dev_global_extrinsic[3],
                                                    dev_global_extrinsic[4] * camera_coord_point_a.x + dev_global_extrinsic[5] * camera_coord_point_a.y + dev_global_extrinsic[6] * camera_coord_point_a.z + dev_global_extrinsic[7],
                                                    dev_global_extrinsic[8] * camera_coord_point_a.x + dev_global_extrinsic[9] * camera_coord_point_a.y + dev_global_extrinsic[10] * camera_coord_point_a.z + dev_global_extrinsic[11]);

                float3 camera_coord_point_b = make_float3(2 * (dev_inv_cam_intrinsic[0] * dev_cam_intrinsic[5] + dev_inv_cam_intrinsic[1] * dev_cam_intrinsic[2] + dev_inv_cam_intrinsic[2] * 1.f),
                                                          2 * (dev_inv_cam_intrinsic[3] * dev_cam_intrinsic[5] + dev_inv_cam_intrinsic[4] * dev_cam_intrinsic[2] + dev_inv_cam_intrinsic[5] * 1.f),
                                                          2 * (dev_inv_cam_intrinsic[6] * dev_cam_intrinsic[5] + dev_inv_cam_intrinsic[7] * dev_cam_intrinsic[2] + dev_inv_cam_intrinsic[8] * 1.f));
                float3 global_point_b = make_float3(dev_global_extrinsic[0] * camera_coord_point_b.x + dev_global_extrinsic[1] * camera_coord_point_b.y + dev_global_extrinsic[2] * camera_coord_point_b.z + dev_global_extrinsic[3],
                                                    dev_global_extrinsic[4] * camera_coord_point_b.x + dev_global_extrinsic[5] * camera_coord_point_b.y + dev_global_extrinsic[6] * camera_coord_point_b.z + dev_global_extrinsic[7],
                                                    dev_global_extrinsic[8] * camera_coord_point_b.x + dev_global_extrinsic[9] * camera_coord_point_b.y + dev_global_extrinsic[10] * camera_coord_point_b.z + dev_global_extrinsic[11]);

                float3 ray_direction_from_center = make_float3(global_point_b.x - global_point_a.x, global_point_b.y - global_point_a.y, global_point_b.z - global_point_a.z);
                float rdfc_magnitude = ray_direction_from_center.x * translation_diff.x + ray_direction_from_center.y * translation_diff.y + ray_direction_from_center.z * translation_diff.z;
                if (rdfc_magnitude > 0)
                {
                    float translation_magnitude = sqrtf(translation_diff.x * translation_diff.x + translation_diff.y * translation_diff.y + translation_diff.z * translation_diff.z);

                    // compute surface point's depth value in global coordinate
                    float3 predicted_3D_point = make_float3(dev_vertices_z[(local_projected_pixel_v - 1) * WIDTH + (local_projected_pixel_u - 1)] * lambda_x,
                                                            dev_vertices_z[(local_projected_pixel_v - 1) * WIDTH + (local_projected_pixel_u - 1)] * lambda_y,
                                                            dev_vertices_z[(local_projected_pixel_v - 1) * WIDTH + (local_projected_pixel_u - 1)] * lambda_z);

                    float3 predicted_global_point = make_float3(dev_global_extrinsic[0] * predicted_3D_point.x + dev_global_extrinsic[1] * predicted_3D_point.y + dev_global_extrinsic[2] * predicted_3D_point.z + dev_global_extrinsic[3],
                                                                dev_global_extrinsic[4] * predicted_3D_point.x + dev_global_extrinsic[5] * predicted_3D_point.y + dev_global_extrinsic[6] * predicted_3D_point.z + dev_global_extrinsic[7],
                                                                dev_global_extrinsic[8] * predicted_3D_point.x + dev_global_extrinsic[9] * predicted_3D_point.y + dev_global_extrinsic[10] * predicted_3D_point.z + dev_global_extrinsic[11]);

                    float3 relative_point_from_camera = make_float3(predicted_global_point.x - dev_global_extrinsic[3],
                                                                    predicted_global_point.y - dev_global_extrinsic[7],
                                                                    predicted_global_point.z - dev_global_extrinsic[11]);

                    float relative_point_distance = sqrtf(relative_point_from_camera.x * relative_point_from_camera.x +
                                                          relative_point_from_camera.y * relative_point_from_camera.y +
                                                          relative_point_from_camera.z * relative_point_from_camera.z);

                    float predicted_depth = relative_point_distance / ray_direction_magnitude;
                    float frk = predicted_depth - ((1.f / ray_direction_magnitude) * translation_magnitude);

                    // update truncated signed distance value for the voxel
                    if (initial_tsdf_construct == true)
                    {
                        if (frk > 0 && frk <= sdf_maximum)
                        {
                            dev_global_tsdf[x] = fminf(1.f, (frk / truncated_distance));
                            dev_global_weight_tsdf[x] = WRK;
                        }
                        else if (frk <= 0 && frk >= sdf_minimum)
                        {
                            dev_global_tsdf[x] = fmaxf(-1.f, frk / truncated_distance);
                            dev_global_weight_tsdf[x] = WRK;
                        }
                    }
                    else
                    {
                        if (frk > 0 && frk <= sdf_maximum)
                        {
                            if (isnan(dev_global_tsdf[x]))
                                dev_global_tsdf[x] = 0.0f;
                            if (isnan(dev_global_weight_tsdf[x]))
                                dev_global_weight_tsdf[x] = 0.0f;

                            frk = fminf(1.f, (frk / truncated_distance));
                            dev_global_tsdf[x] = (dev_global_weight_tsdf[x] * dev_global_tsdf[x] + WRK * frk) / (dev_global_weight_tsdf[x] + WRK);
                            dev_global_weight_tsdf[x] = dev_global_weight_tsdf[x] + WRK;
                        }
                        else if (frk <= 0 && frk >= sdf_minimum)
                        {
                            if (isnan(dev_global_tsdf[x]))
                                dev_global_tsdf[x] = 0.0f;
                            if (isnan(dev_global_weight_tsdf[x]))
                                dev_global_weight_tsdf[x] = 0.0f;

                            frk = fmaxf(-1.f, frk / truncated_distance);
                            dev_global_tsdf[x] = (dev_global_weight_tsdf[x] * dev_global_tsdf[x] + WRK * frk) / (dev_global_weight_tsdf[x] + WRK);
                            dev_global_weight_tsdf[x] = dev_global_weight_tsdf[x] + WRK;
                        }
                    }
                }
            }
        }
    }
}

extern "C" void Projective_TSDF(const int &HEIGHT, const int &WIDTH,
                                const float *voxel_grid_x, const float *voxel_grid_y, const float *voxel_grid_z,
                                const cv::Mat &cam_intrinsic_cv, cv::Mat &global_extrinsic_cv, float *global_tsdf, float *global_weight_tsdf,
                                const cv::Mat &vertices_z_cv, const float &truncated_distance, const float &sdf_minimum, const float &sdf_maximum,
                                const int &voxel_length, const int &voxel_width, const int &voxel_height, bool initial_tsdf_construct)
{
    cv::Mat inv_camera_intrinsic_m = cam_intrinsic_cv.inv();
    cv::Mat inv_extrinsic_matrix = global_extrinsic_cv.inv();

    // gpu data allocation
    float *dev_vertices_z = 0;
    float *dev_voxel_grid_x = 0, *dev_voxel_grid_y = 0, *dev_voxel_grid_z = 0;
    float *dev_global_tsdf = 0;
    float *dev_global_weight_tsdf = 0;

    float *dev_cam_intrinsic = 0, *dev_inv_cam_intrinsic = 0;
    float *dev_global_extrinsic = 0, *dev_inv_global_extrinsic = 0;

    // allocate gpu buffers for data
    cudaMalloc((void **)&dev_vertices_z, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&dev_voxel_grid_x, voxel_length * voxel_width * voxel_height * sizeof(float));
    cudaMalloc((void **)&dev_voxel_grid_y, voxel_length * voxel_width * voxel_height * sizeof(float));
    cudaMalloc((void **)&dev_voxel_grid_z, voxel_length * voxel_width * voxel_height * sizeof(float));
    cudaMalloc((void **)&dev_global_tsdf, voxel_length * voxel_width * voxel_height * sizeof(float));
    cudaMalloc((void **)&dev_global_weight_tsdf, voxel_length * voxel_width * voxel_height * sizeof(float));

    cudaMalloc((void **)&dev_cam_intrinsic, INTRINSIC_ELEMENTS * sizeof(float));
    cudaMalloc((void **)&dev_global_extrinsic, EXTRINSIC_ELEMENTS * sizeof(float));
    cudaMalloc((void **)&dev_inv_cam_intrinsic, INTRINSIC_ELEMENTS * sizeof(float));
    cudaMalloc((void **)&dev_inv_global_extrinsic, EXTRINSIC_ELEMENTS * sizeof(float));

    // copy input from host memory to GPU buffers.
    cudaMemcpy(dev_vertices_z, vertices_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_voxel_grid_x, voxel_grid_x, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_voxel_grid_y, voxel_grid_y, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_voxel_grid_z, voxel_grid_z, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_global_tsdf, global_tsdf, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_global_weight_tsdf, global_weight_tsdf, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_cam_intrinsic, cam_intrinsic_cv.data, INTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_global_extrinsic, global_extrinsic_cv.data, EXTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_inv_cam_intrinsic, inv_camera_intrinsic_m.data, INTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_inv_global_extrinsic, inv_extrinsic_matrix.data, EXTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 64;
    int blocks_per_grid = (voxel_length * voxel_width * voxel_height + threads_per_block - 1) / threads_per_block;

    Projective_TSDF<<<blocks_per_grid, threads_per_block>>>(dev_cam_intrinsic, dev_global_extrinsic,
                                                            dev_inv_cam_intrinsic, dev_inv_global_extrinsic, dev_vertices_z,
                                                            dev_voxel_grid_x, dev_voxel_grid_y, dev_voxel_grid_z,
                                                            dev_global_tsdf, dev_global_weight_tsdf,
                                                            truncated_distance, sdf_minimum, sdf_maximum, initial_tsdf_construct,
                                                            voxel_length, voxel_width, voxel_height,
                                                            HEIGHT, WIDTH);
    cudaDeviceSynchronize();

    // copy output vector from GPU buffer to host memory.
    cudaMemcpy(global_tsdf, dev_global_tsdf, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_weight_tsdf, dev_global_weight_tsdf, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyDeviceToHost);

    // free Device array
    cudaFree(dev_vertices_z);
    cudaFree(dev_voxel_grid_x);
    cudaFree(dev_voxel_grid_y);
    cudaFree(dev_voxel_grid_z);
    cudaFree(dev_global_tsdf);
    cudaFree(dev_global_weight_tsdf);

    cudaFree(dev_cam_intrinsic);
    cudaFree(dev_global_extrinsic);
    cudaFree(dev_inv_cam_intrinsic);
    cudaFree(dev_inv_global_extrinsic);
}