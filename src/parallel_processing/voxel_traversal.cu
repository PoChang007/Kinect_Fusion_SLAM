#include "parallel_processing.h"
#include "cuda_functions.cuh"

__device__ float Ray_Box_Intersection(bool &flag, float3 &ray_direction, float3 &voxel_volume_min, float3 &voxel_volume_max,
                                      float &ray_skipping_global_x, float &ray_skipping_global_y, float &ray_skipping_global_z)
{
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    if (ray_direction.x >= 0)
    {
        tmin = (voxel_volume_min.x - ray_skipping_global_x) / ray_direction.x;
        tmax = (voxel_volume_max.x - ray_skipping_global_x) / ray_direction.x;
    }
    else
    {
        tmin = (voxel_volume_max.x - ray_skipping_global_x) / ray_direction.x;
        tmax = (voxel_volume_min.x - ray_skipping_global_x) / ray_direction.x;
    }

    if (ray_direction.y >= 0)
    {
        tymin = (voxel_volume_min.y - ray_skipping_global_y) / ray_direction.y;
        tymax = (voxel_volume_max.y - ray_skipping_global_y) / ray_direction.y;
    }
    else
    {
        tymin = (voxel_volume_max.y - ray_skipping_global_y) / ray_direction.y;
        tymax = (voxel_volume_min.y - ray_skipping_global_y) / ray_direction.y;
    }

    if ((tmin > tymax) || (tymin > tmax))
    {
        flag = false;
        tmin = -1;
    }
    else
    {
        if (tymin > tmin)
            tmin = tymin;

        if (tymax < tmax)
            tmax = tymax;

        if (ray_direction.z >= 0)
        {
            tzmin = (voxel_volume_min.z - ray_skipping_global_z) / ray_direction.z;
            tzmax = (voxel_volume_max.z - ray_skipping_global_z) / ray_direction.z;
        }
        else
        {
            tzmin = (voxel_volume_max.z - ray_skipping_global_z) / ray_direction.z;
            tzmax = (voxel_volume_min.z - ray_skipping_global_z) / ray_direction.z;
        }

        if ((tmin > tzmax) || (tzmin > tmax))
        {
            flag = false;
            tmin = -1;
        }
        else
        {
            if (tzmin > tmin)
                tmin = tzmin;

            if (tzmax < tmax)
                tmax = tzmax;

            flag = true;
        }
    }

    return tmin;
}

__global__ void Voxel_Traversal(float *dev_surface_points_x, float *dev_surface_points_y, float *dev_surface_points_z,
                                float *dev_surface_normals_x, float *dev_surface_normals_y, float *dev_surface_normals_z,
                                const float *dev_depth_image_coord_x, const float *dev_depth_image_coord_y,
                                const float *dev_cam_intrinsic, float *dev_global_extrinsic,
                                const float *dev_inv_cam_intrinsic, float *dev_inv_global_extrinsic, const float *dev_depth_image,
                                const float *dev_voxel_grid_x, const float *dev_voxel_grid_y,
                                const float *dev_voxel_grid_z, float *dev_global_tsdf,
                                const int voxel_length, const int voxel_width, const int voxel_height,
                                const float truncated_distance, float3 voxel_volume_min, float3 voxel_volume_max, const float voxel_distance,
                                const int HEIGHT, const int WIDTH)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= WIDTH * HEIGHT)
        return;

    if (dev_depth_image[x] != 0)
    {
        // scale the measurement along the pixel ray
        float lambda_x = dev_inv_cam_intrinsic[0] * (dev_depth_image_coord_x[x] + 1.0f) + dev_inv_cam_intrinsic[1] * (dev_depth_image_coord_y[x] + 1.0f) + dev_inv_cam_intrinsic[2] * 1.f;
        float lambda_y = dev_inv_cam_intrinsic[3] * (dev_depth_image_coord_x[x] + 1.0f) + dev_inv_cam_intrinsic[4] * (dev_depth_image_coord_y[x] + 1.0f) + dev_inv_cam_intrinsic[5] * 1.f;
        float lambda_z = dev_inv_cam_intrinsic[6] * (dev_depth_image_coord_x[x] + 1.0f) + dev_inv_cam_intrinsic[7] * (dev_depth_image_coord_y[x] + 1.0f) + dev_inv_cam_intrinsic[8] * 1.f;
        float3 lambda_vector = make_float3(lambda_x, lambda_y, lambda_z);
        // L2 norm operation
        float lambda = sqrtf(lambda_vector.x * lambda_vector.x + lambda_vector.y * lambda_vector.y + lambda_vector.z * lambda_vector.z);
        // use Ray skipping method
        float ray_skipping_local_x = ((lambda * dev_depth_image[x] - truncated_distance) / lambda) * lambda_vector.x;
        float ray_skipping_local_y = ((lambda * dev_depth_image[x] - truncated_distance) / lambda) * lambda_vector.y;
        float ray_skipping_local_z = ((lambda * dev_depth_image[x] - truncated_distance) / lambda) * lambda_vector.z;
        // get start point after ray skipping
        float ray_skipping_global_x = dev_global_extrinsic[0] * ray_skipping_local_x + dev_global_extrinsic[1] * ray_skipping_local_y + dev_global_extrinsic[2] * ray_skipping_local_z + dev_global_extrinsic[3] * 1.f;
        float ray_skipping_global_y = dev_global_extrinsic[4] * ray_skipping_local_x + dev_global_extrinsic[5] * ray_skipping_local_y + dev_global_extrinsic[6] * ray_skipping_local_z + dev_global_extrinsic[7] * 1.f;
        float ray_skipping_global_z = dev_global_extrinsic[8] * ray_skipping_local_x + dev_global_extrinsic[9] * ray_skipping_local_y + dev_global_extrinsic[10] * ray_skipping_local_z + dev_global_extrinsic[11] * 1.f;
        // get ray direction
        float3 camera_position = make_float3(dev_global_extrinsic[3], dev_global_extrinsic[7], dev_global_extrinsic[11]);
        float ray_lambda_global_x = dev_global_extrinsic[0] * lambda_vector.x + dev_global_extrinsic[1] * lambda_vector.y + dev_global_extrinsic[2] * lambda_vector.z + dev_global_extrinsic[3] * 1.f;
        float ray_lambda_global_y = dev_global_extrinsic[4] * lambda_vector.x + dev_global_extrinsic[5] * lambda_vector.y + dev_global_extrinsic[6] * lambda_vector.z + dev_global_extrinsic[7] * 1.f;
        float ray_lambda_global_z = dev_global_extrinsic[8] * lambda_vector.x + dev_global_extrinsic[9] * lambda_vector.y + dev_global_extrinsic[10] * lambda_vector.z + dev_global_extrinsic[11] * 1.f;

        float3 ray_direction = make_float3((ray_lambda_global_x - camera_position.x), (ray_lambda_global_y - camera_position.y), (ray_lambda_global_z - camera_position.z));
        Get_Normalized(ray_direction);

        // ray box intersection test
        bool flag = false;
        float tmin = Ray_Box_Intersection(flag, ray_direction, voxel_volume_min, voxel_volume_max,
                                          ray_skipping_global_x, ray_skipping_global_y, ray_skipping_global_z);

        if (flag == true)
        {
            if (tmin < 0)
                tmin = 0;

            float3 start = make_float3((ray_skipping_global_x + tmin * ray_direction.x),
                                       (ray_skipping_global_y + tmin * ray_direction.y),
                                       (ray_skipping_global_z + tmin * ray_direction.z));
            float3 box_size = make_float3((voxel_volume_max.x - voxel_volume_min.x),
                                          (voxel_volume_max.y - voxel_volume_min.y),
                                          (voxel_volume_max.z - voxel_volume_min.z));
            // starting point in voxel indices
            int voxel_index_x = (int)floor(((start.x - voxel_volume_min.x) / box_size.x) * voxel_length) + 1;
            int voxel_index_y = (int)floor(((start.y - voxel_volume_min.y) / box_size.y) * voxel_width) + 1;
            int voxel_index_z = (int)floor(((start.z - voxel_volume_min.z) / box_size.z) * voxel_height) + 1;
            int XY, XZ, YX, YZ, ZX, ZY;

            // if value is on maximum edge
            if (voxel_index_x == (voxel_length + 1))
                voxel_index_x = voxel_index_x - 1;

            if (voxel_index_y == (voxel_width + 1))
                voxel_index_y = voxel_index_y - 1;

            if (voxel_index_z == (voxel_height + 1))
                voxel_index_z = voxel_index_z - 1;

            float t_voxel_x, t_voxel_y, t_voxel_z;
            float step_x, step_y, step_z;

            if (ray_direction.x >= 0)
            {
                t_voxel_x = (float)(voxel_index_x) / voxel_length;
                step_x = 1;
            }
            else
            {
                t_voxel_x = (float)(voxel_index_x - 1) / voxel_length;
                step_x = -1;
            }

            if (ray_direction.y >= 0)
            {
                t_voxel_y = (float)(voxel_index_y) / voxel_width;
                step_y = 1;
            }
            else
            {
                t_voxel_y = (float)(voxel_index_y - 1) / voxel_width;
                step_y = -1;
            }

            if (ray_direction.z >= 0)
            {
                t_voxel_z = (float)(voxel_index_z) / voxel_height;
                step_z = 1;
            }
            else
            {
                t_voxel_z = (float)(voxel_index_z - 1) / voxel_height;
                step_z = -1;
            }

            // first Voxel(x,y,z) is rightmost if direction is positive, leftmost if the direction is negative
            // get the first traversal voxel
            float3 voxel_max = make_float3(voxel_volume_min.x + t_voxel_x * box_size.x,
                                           voxel_volume_min.y + t_voxel_y * box_size.y,
                                           voxel_volume_min.z + t_voxel_z * box_size.z);
            // time in first Voxel(x, y, Z)
            float3 t_max = make_float3(tmin + ((voxel_max.x - start.x) / ray_direction.x),
                                       tmin + ((voxel_max.y - start.y) / ray_direction.y),
                                       tmin + ((voxel_max.z - start.z) / ray_direction.z));

            float3 voxel_size = make_float3((float)voxel_length, (float)voxel_width, (float)voxel_height);
            // time for ray pass through each voxel
            float3 t_delta = make_float3(voxel_distance / abs(ray_direction.x), voxel_distance / abs(ray_direction.y), voxel_distance / abs(ray_direction.z));

            float traversal_recording_tsdf = 0.0f;
            // match matlab index
            int voxel_index = (voxel_index_x - 1) * voxel_width * voxel_height + (voxel_index_y - 1) * voxel_height + (voxel_index_z - 1);

            // check if the ray is inside global 3D volume
            while (voxel_index_x < (voxel_size.x - 1) && voxel_index_x > 0 &&
                   voxel_index_y < (voxel_size.y - 1) && voxel_index_y > 0 &&
                   voxel_index_z < (voxel_size.z - 1) && voxel_index_z > 0)
            {
                if (isnan(dev_global_tsdf[voxel_index]) && traversal_recording_tsdf < 0.f)
                {
                    break;
                }
                if (dev_global_tsdf[voxel_index] < 0.f && isnan(traversal_recording_tsdf))
                {
                    break;
                }
                // check if back face is found
                if (dev_global_tsdf[voxel_index] > 0.f && traversal_recording_tsdf < 0.f)
                {
                    break;
                }

                // check the zero crossing region
                if (dev_global_tsdf[voxel_index] < 0.f && traversal_recording_tsdf > 0.f)
                {
                    // zero crossing
                    // match matlab index
                    int x_n_index = ((voxel_index_x - 1) - 1);
                    int x_p_index = ((voxel_index_x - 1) + 1);
                    int y_n_index = ((voxel_index_y - 1) - 1);
                    int y_p_index = ((voxel_index_y - 1) + 1);
                    int z_n_index = ((voxel_index_z - 1) - 1);
                    int z_p_index = ((voxel_index_z - 1) + 1);

                    if (x_n_index >= voxel_length || x_n_index < 0)
                    {
                        break;
                    }
                    if (x_p_index >= voxel_length || x_p_index < 0)
                    {
                        break;
                    }
                    if ((XY - 1) >= voxel_length || (XY - 1) < 0)
                    {
                        break;
                    }
                    if ((XZ - 1) >= voxel_length || (XZ - 1) < 0)
                    {
                        break;
                    }

                    if (y_n_index >= voxel_width || y_n_index < 0)
                    {
                        break;
                    }
                    if (y_p_index >= voxel_width || y_p_index < 0)
                    {
                        break;
                    }
                    if ((YX - 1) >= voxel_width || (YX - 1) < 0)
                    {
                        break;
                    }
                    if ((YZ - 1) >= voxel_width || (YZ - 1) < 0)
                    {
                        break;
                    }

                    if (z_n_index >= voxel_height || z_n_index < 0)
                    {
                        break;
                    }
                    if (z_p_index >= voxel_height || z_p_index < 0)
                    {
                        break;
                    }
                    if ((ZX - 1) >= voxel_height || (ZX - 1) < 0)
                    {
                        break;
                    }
                    if ((ZY - 1) >= voxel_height || (ZY - 1) < 0)
                    {
                        break;
                    }

                    // match matlab index
                    // for position interpolation index
                    int x_np = x_n_index * voxel_width * voxel_height + (XY - 1) * voxel_height + XZ - 1;
                    int x_pp = x_p_index * voxel_width * voxel_height + (XY - 1) * voxel_height + XZ - 1;
                    int y_np = (YX - 1) * voxel_width * voxel_height + y_n_index * voxel_height + YZ - 1;
                    int y_pp = (YX - 1) * voxel_width * voxel_height + y_p_index * voxel_height + YZ - 1;
                    int z_np = (ZX - 1) * voxel_width * voxel_height + (ZY - 1) * voxel_height + z_n_index;
                    int z_pp = (ZX - 1) * voxel_width * voxel_height + (ZY - 1) * voxel_height + z_p_index;
                    // for normal interpolation index
                    int x_nn = (voxel_index_x - 2) * voxel_width * voxel_height + (voxel_index_y - 1) * voxel_height + (voxel_index_z - 1);
                    int x_pn = (voxel_index_x)*voxel_width * voxel_height + (voxel_index_y - 1) * voxel_height + (voxel_index_z - 1);
                    int y_nn = (voxel_index_x - 1) * voxel_width * voxel_height + (voxel_index_y - 2) * voxel_height + (voxel_index_z - 1);
                    int y_pn = (voxel_index_x - 1) * voxel_width * voxel_height + (voxel_index_y)*voxel_height + (voxel_index_z - 1);
                    int z_nn = (voxel_index_x - 1) * voxel_width * voxel_height + (voxel_index_y - 1) * voxel_height + (voxel_index_z - 2);
                    int z_pn = (voxel_index_x - 1) * voxel_width * voxel_height + (voxel_index_y - 1) * voxel_height + (voxel_index_z);

                    // check nan for XY, XZ...
                    if (isnan(dev_global_tsdf[x_np]) || isnan(dev_global_tsdf[x_pp]) ||
                        isnan(dev_global_tsdf[y_np]) || isnan(dev_global_tsdf[y_pp]) ||
                        isnan(dev_global_tsdf[z_np]) || isnan(dev_global_tsdf[z_pp]) || isnan(dev_global_tsdf[voxel_index]))
                    {
                        break;
                    }

                    // check nan for x-1, x+1, y-1, y+1...
                    if (isnan(dev_global_tsdf[x_nn]) || isnan(dev_global_tsdf[x_pn]) ||
                        isnan(dev_global_tsdf[y_nn]) || isnan(dev_global_tsdf[y_pn]) ||
                        isnan(dev_global_tsdf[z_nn]) || isnan(dev_global_tsdf[z_pn]))
                    {
                        break;
                    }

                    float3 surface_prediction;
                    float3 surface_normal_prediction;
                    if (ray_direction.x > 0)
                    {
                        float sample_point_1 = dev_voxel_grid_x[x_np];
                        surface_prediction.x = sample_point_1 + voxel_distance * (abs(dev_global_tsdf[x_np]) / (abs(dev_global_tsdf[x_np] - dev_global_tsdf[voxel_index])));
                    }
                    else
                    {
                        float sample_point_1 = dev_voxel_grid_x[x_pp];
                        surface_prediction.x = sample_point_1 - voxel_distance * (abs(dev_global_tsdf[x_pp]) / (abs(dev_global_tsdf[x_pp] - dev_global_tsdf[voxel_index])));
                    }

                    if (ray_direction.y > 0)
                    {
                        float sample_point_1 = dev_voxel_grid_y[y_np];
                        surface_prediction.y = sample_point_1 + voxel_distance * (abs(dev_global_tsdf[y_np]) / (abs(dev_global_tsdf[y_np] - dev_global_tsdf[voxel_index])));
                    }
                    else
                    {
                        float sample_point_1 = dev_voxel_grid_y[y_pp];
                        surface_prediction.y = sample_point_1 - voxel_distance * (abs(dev_global_tsdf[y_pp]) / (abs(dev_global_tsdf[y_pp] - dev_global_tsdf[voxel_index])));
                    }

                    if (ray_direction.z > 0)
                    {
                        float sample_point_1 = dev_voxel_grid_z[z_np];
                        surface_prediction.z = sample_point_1 + voxel_distance * (abs(dev_global_tsdf[z_np]) / (abs(dev_global_tsdf[z_np] - dev_global_tsdf[voxel_index])));
                    }
                    else
                    {
                        float sample_point_1 = dev_voxel_grid_z[z_pp];
                        surface_prediction.z = sample_point_1 - voxel_distance * (abs(dev_global_tsdf[z_pp]) / (abs(dev_global_tsdf[z_pp] - dev_global_tsdf[voxel_index])));
                    }

                    surface_normal_prediction.x = (dev_global_tsdf[x_pn] - dev_global_tsdf[x_nn]) / 2.0f;
                    surface_normal_prediction.y = (dev_global_tsdf[y_pn] - dev_global_tsdf[y_nn]) / 2.0f;
                    surface_normal_prediction.z = (dev_global_tsdf[z_pn] - dev_global_tsdf[z_nn]) / 2.0f;

                    //check the normal vector to make sure pointing outward
                    float3 vector_a = make_float3((camera_position.x - surface_prediction.x), (camera_position.y - surface_prediction.y), (camera_position.z - surface_prediction.z));
                    Get_Normalized(vector_a);
                    float3 vector_b = make_float3(surface_normal_prediction.x, surface_normal_prediction.y, surface_normal_prediction.z);
                    Get_Normalized(vector_b);

                    double cos_theta = vector_a.x * vector_b.x + vector_a.y * vector_b.y + vector_a.z * vector_b.z;
                    double angle = acos(cos_theta) * (180.0f / CUDART_PI_F);

                    if (angle > 90)
                    {
                        vector_b.x = -vector_b.x;
                        vector_b.y = -vector_b.y;
                        vector_b.z = -vector_b.z;
                    }

                    dev_surface_points_x[x] = surface_prediction.x;
                    dev_surface_points_y[x] = surface_prediction.y;
                    dev_surface_points_z[x] = surface_prediction.z;
                    dev_surface_normals_x[x] = vector_b.x;
                    dev_surface_normals_y[x] = vector_b.y;
                    dev_surface_normals_z[x] = vector_b.z;
                    break;
                }

                // store the previous voxel indices and time
                traversal_recording_tsdf = dev_global_tsdf[voxel_index];

                // if voxel in x dimension is the fastest to achieve the next voxel
                if (t_max.x < t_max.y)
                {
                    if (t_max.x < t_max.z)
                    {
                        voxel_index_x = voxel_index_x + step_x;
                        XY = voxel_index_y;
                        XZ = voxel_index_z;
                        t_max.x = t_max.x + t_delta.x;
                    }
                    else
                    {
                        voxel_index_z = voxel_index_z + step_z;
                        ZX = voxel_index_x;
                        ZY = voxel_index_y;
                        t_max.z = t_max.z + t_delta.z;
                    }
                }
                else // else test if voxel in y dimension is the fastest to achieve the next voxel
                {
                    if (t_max.y < t_max.z)
                    {
                        voxel_index_y = voxel_index_y + step_y;
                        YX = voxel_index_x;
                        YZ = voxel_index_z;
                        t_max.y = t_max.y + t_delta.y;
                    }
                    else
                    {
                        voxel_index_z = voxel_index_z + step_z;
                        ZX = voxel_index_x;
                        ZY = voxel_index_y;
                        t_max.z = t_max.z + t_delta.z;
                    }
                }
                // update to new voxel
                // match matlab index
                voxel_index = (voxel_index_x - 1) * voxel_width * voxel_height + (voxel_index_y - 1) * voxel_height + (voxel_index_z - 1);
            }
        }
    }
}

extern "C" void Ray_Casting(const int &HEIGHT, const int &WIDTH,
                            cv::Mat &surface_points_x_cv, cv::Mat &surface_points_y_cv, cv::Mat &surface_points_z_cv,
                            cv::Mat &surface_normals_x_cv, cv::Mat &surface_normals_y_cv, cv::Mat &surface_normals_z_cv,
                            const float *voxel_grid_x, const float *voxel_grid_y, const float *voxel_grid_z,
                            const float *depth_image_coord_y_cv, const float *depth_image_coord_x_cv,
                            const cv::Mat &cam_intrinsic_cv, cv::Mat &global_extrinsic_cv, float *global_tsdf,
                            const cv::Mat &vertices_z_cv, cv::Mat &traversal_recording, const float &truncated_distance,
                            const int &voxel_length, const int &voxel_width, const int &voxel_height,
                            const int &voxel_grid_x_start_pos, const int &voxel_grid_y_start_pos, const int &voxel_grid_z_start_pos,
                            const float &voxel_distance)
{
    // get current frame's camera position
    cv::Mat camera_position = global_extrinsic_cv(cv::Range(0, 3), cv::Range(3, 4));
    float3 voxel_volume_min = make_float3((float)(voxel_grid_x_start_pos * voxel_distance), (float)(voxel_grid_y_start_pos * voxel_distance), (float)(voxel_grid_z_start_pos * voxel_distance));
    float voxel_max_x = (float)((voxel_grid_x_start_pos + voxel_length) * voxel_distance);
    float voxel_max_y = (float)((voxel_grid_y_start_pos + voxel_width) * voxel_distance);
    float voxel_max_z = (float)((voxel_grid_z_start_pos + voxel_height) * voxel_distance);
    float3 voxel_volume_max = make_float3(voxel_max_x, voxel_max_y, voxel_max_z);

    cv::Mat inv_camera_intrinsic_m = cam_intrinsic_cv.inv();
    cv::Mat inv_extrinsic_matrix = global_extrinsic_cv.inv();

    // gpu data allocation
    float *dev_surface_points_x = 0, *dev_surface_points_y = 0, *dev_surface_points_z = 0;
    float *dev_surface_normals_x = 0, *dev_surface_normals_y = 0, *dev_surface_normals_z = 0;

    float *dev_depth_image_coord_y = 0, *dev_depth_image_coord_x = 0;
    float *dev_cam_intrinsic = 0, *dev_inv_cam_intrinsic = 0;
    float *dev_global_extrinsic = 0, *dev_inv_global_extrinsic = 0;

    float *dev_vertices_z = 0;
    float *dev_voxel_grid_x = 0, *dev_voxel_grid_y = 0, *dev_voxel_grid_z = 0;
    float *dev_global_tsdf = 0;

    // allocate GPU buffers for data
    cudaMalloc((void **)&dev_surface_points_x, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&dev_surface_points_y, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&dev_surface_points_z, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&dev_surface_normals_x, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&dev_surface_normals_y, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&dev_surface_normals_z, WIDTH * HEIGHT * sizeof(float));

    cudaMalloc((void **)&dev_depth_image_coord_x, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&dev_depth_image_coord_y, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&dev_cam_intrinsic, INTRINSIC_ELEMENTS * sizeof(float));
    cudaMalloc((void **)&dev_global_extrinsic, EXTRINSIC_ELEMENTS * sizeof(float));
    cudaMalloc((void **)&dev_inv_cam_intrinsic, INTRINSIC_ELEMENTS * sizeof(float));
    cudaMalloc((void **)&dev_inv_global_extrinsic, EXTRINSIC_ELEMENTS * sizeof(float));

    cudaMalloc((void **)&dev_vertices_z, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void **)&dev_voxel_grid_x, voxel_length * voxel_width * voxel_height * sizeof(float));
    cudaMalloc((void **)&dev_voxel_grid_y, voxel_length * voxel_width * voxel_height * sizeof(float));
    cudaMalloc((void **)&dev_voxel_grid_z, voxel_length * voxel_width * voxel_height * sizeof(float));
    cudaMalloc((void **)&dev_global_tsdf, voxel_length * voxel_width * voxel_height * sizeof(float));

    // copy input from host memory to GPU buffers.
    cudaMemcpy(dev_surface_points_x, surface_points_x_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_surface_points_y, surface_points_y_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_surface_points_z, surface_points_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_surface_normals_x, surface_normals_x_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_surface_normals_y, surface_normals_y_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_surface_normals_z, surface_normals_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_depth_image_coord_y, depth_image_coord_y_cv, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_depth_image_coord_x, depth_image_coord_x_cv, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cam_intrinsic, cam_intrinsic_cv.data, INTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_global_extrinsic, global_extrinsic_cv.data, EXTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_inv_cam_intrinsic, inv_camera_intrinsic_m.data, INTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_inv_global_extrinsic, inv_extrinsic_matrix.data, EXTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_vertices_z, vertices_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_voxel_grid_x, voxel_grid_x, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_voxel_grid_y, voxel_grid_y, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_voxel_grid_z, voxel_grid_z, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_global_tsdf, global_tsdf, voxel_length * voxel_width * voxel_height * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 64;
    int blocks_per_grid = (WIDTH * HEIGHT + threads_per_block - 1) / threads_per_block;

    Voxel_Traversal<<<blocks_per_grid, threads_per_block>>>(dev_surface_points_x, dev_surface_points_y, dev_surface_points_z,
                                                            dev_surface_normals_x, dev_surface_normals_y, dev_surface_normals_z,
                                                            dev_depth_image_coord_x, dev_depth_image_coord_y,
                                                            dev_cam_intrinsic, dev_global_extrinsic,
                                                            dev_inv_cam_intrinsic, dev_inv_global_extrinsic, dev_vertices_z,
                                                            dev_voxel_grid_x, dev_voxel_grid_y, dev_voxel_grid_z, dev_global_tsdf,
                                                            voxel_length, voxel_width, voxel_height,
                                                            truncated_distance, voxel_volume_min, voxel_volume_max, voxel_distance,
                                                            HEIGHT, WIDTH);
    cudaDeviceSynchronize();

    // copy output vector from GPU buffer to host memory
    cudaMemcpy(surface_points_x_cv.data, dev_surface_points_x, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(surface_points_y_cv.data, dev_surface_points_y, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(surface_points_z_cv.data, dev_surface_points_z, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(surface_normals_x_cv.data, dev_surface_normals_x, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(surface_normals_y_cv.data, dev_surface_normals_y, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(surface_normals_z_cv.data, dev_surface_normals_z, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    // free Device array
    cudaFree(dev_surface_points_x);
    cudaFree(dev_surface_points_y);
    cudaFree(dev_surface_points_z);
    cudaFree(dev_surface_normals_x);
    cudaFree(dev_surface_normals_y);
    cudaFree(dev_surface_normals_z);

    cudaFree(dev_depth_image_coord_y);
    cudaFree(dev_depth_image_coord_x);
    cudaFree(dev_cam_intrinsic);
    cudaFree(dev_global_extrinsic);
    cudaFree(dev_inv_cam_intrinsic);
    cudaFree(dev_inv_global_extrinsic);

    cudaFree(dev_vertices_z);
    cudaFree(dev_voxel_grid_x);
    cudaFree(dev_voxel_grid_y);
    cudaFree(dev_voxel_grid_z);
    cudaFree(dev_global_tsdf);
}