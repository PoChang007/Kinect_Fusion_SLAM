#include "kinfu_pipeline.h"

__global__ void Calculate_Vertices_And_Normals(float *dev_vertices_z, float *dev_inv_cam_intrinsic,
											  float *dev_vertices_x, float *dev_vertices_y,
											  float *dev_normals_x, float *dev_normals_y, float *dev_normals_z,
											  uint8_t *dev_vertex_mask, float *dev_depth_image_coord_y, float *dev_depth_image_coord_x,
											  const int HEIGHT, const int WIDTH)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= WIDTH * HEIGHT)
		return;

	int current_y = dev_depth_image_coord_y[x];
	int current_x = dev_depth_image_coord_x[x];

	// Get rid of the last row/col
	if (dev_vertices_z[x] != 0 && current_y != (HEIGHT - 1) && current_x != (WIDTH - 1))
	{
		dev_vertex_mask[x] = 1;
		int next_row = (current_y + 1) * WIDTH + (current_x);
		int next_col = (current_y)*WIDTH + (current_x + 1);

		if (next_row > 0 && next_col > 0 && dev_vertices_z[next_row] != 0 && dev_vertices_z[next_col] != 0)
		{
			float raw_vertices_x = dev_vertices_z[x] * (dev_inv_cam_intrinsic[0] * (float)(current_x + 1) + dev_inv_cam_intrinsic[1] * (float)(current_y + 1) + dev_inv_cam_intrinsic[2]);
			float raw_vertices_y = dev_vertices_z[x] * (dev_inv_cam_intrinsic[3] * (float)(current_x + 1) + dev_inv_cam_intrinsic[4] * (float)(current_y + 1) + dev_inv_cam_intrinsic[5]);
			float raw_vertices_x_next_row = dev_vertices_z[next_row] * (dev_inv_cam_intrinsic[0] * (float)(current_x + 1) + dev_inv_cam_intrinsic[1] * (float)(current_y + 2) + dev_inv_cam_intrinsic[2]);
			float raw_vertices_x_next_col = dev_vertices_z[next_col] * (dev_inv_cam_intrinsic[0] * (float)(current_x + 2) + dev_inv_cam_intrinsic[1] * (float)(current_y + 1) + dev_inv_cam_intrinsic[2]);
			float raw_vertices_y_next_row = dev_vertices_z[next_row] * (dev_inv_cam_intrinsic[3] * (float)(current_x + 1) + dev_inv_cam_intrinsic[4] * (float)(current_y + 2) + dev_inv_cam_intrinsic[5]);
			float raw_vertices_y_next_col = dev_vertices_z[next_col] * (dev_inv_cam_intrinsic[3] * (float)(current_x + 2) + dev_inv_cam_intrinsic[4] * (float)(current_y + 1) + dev_inv_cam_intrinsic[5]);

			//cross prodcut
			float nI = (raw_vertices_y_next_row - raw_vertices_y) * (dev_vertices_z[next_col] - dev_vertices_z[x]) -
					   (dev_vertices_z[next_row] - dev_vertices_z[x]) * (raw_vertices_y_next_col - raw_vertices_y);
			float nJ = (dev_vertices_z[next_row] - dev_vertices_z[x]) * (raw_vertices_x_next_col - raw_vertices_x) -
					   (raw_vertices_x_next_row - raw_vertices_x) * (dev_vertices_z[next_col] - dev_vertices_z[x]);
			float nK = (raw_vertices_x_next_row - raw_vertices_x) * (raw_vertices_y_next_col - raw_vertices_y) -
					   (raw_vertices_y_next_row - raw_vertices_y) * (raw_vertices_x_next_col - raw_vertices_x);

			float u_nI = nI / sqrtf(nI * nI + nJ * nJ + nK * nK);
			float u_nJ = nJ / sqrtf(nI * nI + nJ * nJ + nK * nK);
			float u_nK = nK / sqrtf(nI * nI + nJ * nJ + nK * nK);

			dev_vertices_x[x] = raw_vertices_x;
			dev_vertices_y[x] = raw_vertices_y;
			dev_normals_x[x] = u_nI;
			dev_normals_y[x] = u_nJ;
			dev_normals_z[x] = u_nK;
		}
	}
}

extern "C" void Calculate_Vertices_And_Normals(const int HEIGHT, int WIDTH, 
	 										   cv::Mat &vertices_z_cv, cv::Mat &cam_intrinsic_cv,
											   cv::Mat &vertices_x_cv, cv::Mat &vertices_y_cv,
											   cv::Mat &normals_x_cv, cv::Mat &normals_y_cv, cv::Mat &normals_z_cv,
											   cv::Mat &vertex_mask, const float *depth_image_coord_y, const float *depth_image_coord_x)
{
	cv::Mat inv_cam_intrinsic_cv = cam_intrinsic_cv.inv();

	// gpu data allocation
	float *dev_vertices_x = 0, *dev_vertices_y = 0, *dev_vertices_z = 0;
	float *dev_normals_x = 0, *dev_normals_y = 0, *dev_normals_z = 0;

	float *dev_depth_image_coord_x = 0, *dev_depth_image_coord_y = 0;
	uint8_t *dev_vertex_mask = 0;
	float *dev_inv_cam_intrinsic = 0;

	// allocate GPU buffers for data

	cudaMalloc((void **)&dev_vertices_x, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_vertices_y, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_vertices_z, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_normals_x, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_normals_y, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_normals_z, WIDTH * HEIGHT * sizeof(float));

	cudaMalloc((void **)&dev_depth_image_coord_x, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_depth_image_coord_y, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_vertex_mask, WIDTH * HEIGHT * sizeof(uint8_t));
	cudaMalloc((void **)&dev_inv_cam_intrinsic, INTRINSIC_ELEMENTS * sizeof(float));

	// copy input from host memory to GPU buffers
	cudaMemcpy(dev_vertices_x, vertices_x_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertices_y, vertices_y_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertices_z, vertices_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_normals_x, normals_x_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_normals_y, normals_y_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_normals_z, normals_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_depth_image_coord_x, depth_image_coord_x, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_depth_image_coord_y, depth_image_coord_y, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertex_mask, vertex_mask.data, WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inv_cam_intrinsic, inv_cam_intrinsic_cv.data, INTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

	int threads_per_block = 64;
	int blocks_per_grid = (WIDTH * HEIGHT + threads_per_block - 1) / threads_per_block;

	Calculate_Vertices_And_Normals<<<blocks_per_grid, threads_per_block>>>(dev_vertices_z, dev_inv_cam_intrinsic,
																		  dev_vertices_x, dev_vertices_y,
																		  dev_normals_x, dev_normals_y, dev_normals_z,
																		  dev_vertex_mask, dev_depth_image_coord_y, dev_depth_image_coord_x,
																		  HEIGHT, WIDTH);
	cudaDeviceSynchronize();

	// copy output vector from gpu buffer to host memory
	cudaMemcpy(vertices_x_cv.data, dev_vertices_x, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(vertices_y_cv.data, dev_vertices_y, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(normals_x_cv.data, dev_normals_x, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(normals_y_cv.data, dev_normals_y, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(normals_z_cv.data, dev_normals_z, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(vertex_mask.data, dev_vertex_mask, WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	// free Device array
	cudaFree(dev_vertices_x);
	cudaFree(dev_vertices_y);
	cudaFree(dev_vertices_z);
	cudaFree(dev_normals_x);
	cudaFree(dev_normals_y);
	cudaFree(dev_normals_z);

	cudaFree(dev_depth_image_coord_x);
	cudaFree(dev_depth_image_coord_y);
	cudaFree(dev_vertex_mask);
	cudaFree(dev_inv_cam_intrinsic);
}