#include "kinfu_pipeline.h"
#include "cuda_functions.cuh"

#define SIZE_6 6
#define SIZE_36 36

__global__ void ICP(const float *dev_vertices_x, const float *dev_vertices_y, const float *dev_vertices_z,
					const float *dev_normals_x, const float *dev_normals_y, const float *dev_normals_z,
					const float *dev_surface_points_x, const float *dev_surface_points_y, const float *dev_surface_points_z,
					const float *dev_surface_normals_x, const float *dev_surface_normals_y, const float *dev_surface_normals_z,
					const float *dev_inv_cam_intrinsic, float *dev_inv_global_extrinsic,
					float *dev_refinement_6dof_trans, float *dev_frame_to_frame_trans,
					float *dev_linear_system_right_matrix, float *dev_linear_system_left_matrix, const uint8_t *dev_vertex_mask,
					const int HEIGHT, const int WIDTH)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= WIDTH * HEIGHT)
		return;

	if (dev_vertices_z[x] != 0 && dev_vertex_mask[x] != 0)
	{
		float camera_x = dev_frame_to_frame_trans[0] * dev_vertices_x[x] + dev_frame_to_frame_trans[1] * dev_vertices_y[x] + dev_frame_to_frame_trans[2] * dev_vertices_z[x] + dev_frame_to_frame_trans[3];
		float camera_y = dev_frame_to_frame_trans[4] * dev_vertices_x[x] + dev_frame_to_frame_trans[5] * dev_vertices_y[x] + dev_frame_to_frame_trans[6] * dev_vertices_z[x] + dev_frame_to_frame_trans[7];
		float camera_z = dev_frame_to_frame_trans[8] * dev_vertices_x[x] + dev_frame_to_frame_trans[9] * dev_vertices_y[x] + dev_frame_to_frame_trans[10] * dev_vertices_z[x] + dev_frame_to_frame_trans[11];

		float projected_x = dev_inv_cam_intrinsic[0] * camera_x + dev_inv_cam_intrinsic[1] * camera_y + dev_inv_cam_intrinsic[2] * camera_z;
		float projected_y = dev_inv_cam_intrinsic[3] * camera_x + dev_inv_cam_intrinsic[4] * camera_y + dev_inv_cam_intrinsic[5] * camera_z;
		float projected_z = dev_inv_cam_intrinsic[6] * camera_x + dev_inv_cam_intrinsic[7] * camera_y + dev_inv_cam_intrinsic[8] * camera_z;

		int local_projected_pixel_u = (int)roundf(projected_x / projected_z);
		int local_projected_pixel_v = (int)roundf(projected_y / projected_z);

		if (local_projected_pixel_u > 0 && local_projected_pixel_u <= WIDTH)
		{
			if (local_projected_pixel_v > 0 && local_projected_pixel_v <= HEIGHT)
			{
				// T^Z(g, k) * Vk
				// Update vertex for the next round of ICP
				float update_vertex_x = dev_refinement_6dof_trans[0] * dev_vertices_x[x] + dev_refinement_6dof_trans[1] * dev_vertices_y[x] + dev_refinement_6dof_trans[2] * dev_vertices_z[x] + dev_refinement_6dof_trans[3];
				float update_vertex_y = dev_refinement_6dof_trans[4] * dev_vertices_x[x] + dev_refinement_6dof_trans[5] * dev_vertices_y[x] + dev_refinement_6dof_trans[6] * dev_vertices_z[x] + dev_refinement_6dof_trans[7];
				float update_vertex_z = dev_refinement_6dof_trans[8] * dev_vertices_x[x] + dev_refinement_6dof_trans[9] * dev_vertices_y[x] + dev_refinement_6dof_trans[10] * dev_vertices_z[x] + dev_refinement_6dof_trans[11];

				int projected_index = (local_projected_pixel_v - 1) * WIDTH + (local_projected_pixel_u - 1);
				float3 distance_difference = make_float3(update_vertex_x - dev_surface_points_x[projected_index],
														 update_vertex_y - dev_surface_points_y[projected_index],
														 update_vertex_z - dev_surface_points_z[projected_index]);
				float distance = sqrtf(distance_difference.x * distance_difference.x + distance_difference.y * distance_difference.y + distance_difference.z * distance_difference.z);
				if (distance <= 20.0f && distance >= 0.0f)
				{
					// R^Z(g, k) * Nk
					// update normal vector
					float3 update_normal = make_float3(dev_refinement_6dof_trans[0] * dev_normals_x[x] + dev_refinement_6dof_trans[1] * dev_normals_y[x] + dev_refinement_6dof_trans[2] * dev_normals_z[x],
													   dev_refinement_6dof_trans[4] * dev_normals_x[x] + dev_refinement_6dof_trans[5] * dev_normals_y[x] + dev_refinement_6dof_trans[6] * dev_normals_z[x],
													   dev_refinement_6dof_trans[8] * dev_normals_x[x] + dev_refinement_6dof_trans[9] * dev_normals_y[x] + dev_refinement_6dof_trans[10] * dev_normals_z[x]);
					Get_Normalized(update_normal);

					float3 vector_a = make_float3(dev_surface_normals_x[projected_index],
												  dev_surface_normals_y[projected_index],
												  dev_surface_normals_z[projected_index]);

					Get_Normalized(vector_a);

					float cos_theta = (update_normal.x * vector_a.x + update_normal.y * vector_a.y + update_normal.z * vector_a.z);
					float normal_angle = acos(cos_theta) * (180.0f / CUDART_PI_F);

					if (normal_angle < 0)
						printf("negative angle \n");

					if (normal_angle <= 30.0f && normal_angle >= 0.0f)
					{
						// solve 6X6 linear sysem
						float a_transpose_1 = update_vertex_z * dev_surface_normals_y[projected_index] + (-update_vertex_y * dev_surface_normals_z[projected_index]);
						float a_transpose_2 = -update_vertex_z * dev_surface_normals_x[projected_index] + (update_vertex_x * dev_surface_normals_z[projected_index]);
						float a_transpose_3 = update_vertex_y * dev_surface_normals_x[projected_index] + (-update_vertex_x * dev_surface_normals_y[projected_index]);
						float a_transpose_4 = dev_surface_normals_x[projected_index];
						float a_transpose_5 = dev_surface_normals_y[projected_index];
						float a_transpose_6 = dev_surface_normals_z[projected_index];

						float b = dev_surface_normals_x[projected_index] * (dev_surface_points_x[projected_index] - update_vertex_x) +
								  dev_surface_normals_y[projected_index] * (dev_surface_points_y[projected_index] - update_vertex_y) +
								  dev_surface_normals_z[projected_index] * (dev_surface_points_z[projected_index] - update_vertex_z);

						dev_linear_system_right_matrix[x + WIDTH * HEIGHT * 0] = a_transpose_1 * b;
						dev_linear_system_right_matrix[x + WIDTH * HEIGHT * 1] = a_transpose_2 * b;
						dev_linear_system_right_matrix[x + WIDTH * HEIGHT * 2] = a_transpose_3 * b;
						dev_linear_system_right_matrix[x + WIDTH * HEIGHT * 3] = a_transpose_4 * b;
						dev_linear_system_right_matrix[x + WIDTH * HEIGHT * 4] = a_transpose_5 * b;
						dev_linear_system_right_matrix[x + WIDTH * HEIGHT * 5] = a_transpose_6 * b;

						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 0] = a_transpose_1 * a_transpose_1;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 1] = a_transpose_1 * a_transpose_2;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 2] = a_transpose_1 * a_transpose_3;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 3] = a_transpose_1 * a_transpose_4;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 4] = a_transpose_1 * a_transpose_5;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 5] = a_transpose_1 * a_transpose_6;

						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 6] = a_transpose_2 * a_transpose_1;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 7] = a_transpose_2 * a_transpose_2;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 8] = a_transpose_2 * a_transpose_3;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 9] = a_transpose_2 * a_transpose_4;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 10] = a_transpose_2 * a_transpose_5;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 11] = a_transpose_2 * a_transpose_6;

						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 12] = a_transpose_3 * a_transpose_1;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 13] = a_transpose_3 * a_transpose_2;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 14] = a_transpose_3 * a_transpose_3;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 15] = a_transpose_3 * a_transpose_4;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 16] = a_transpose_3 * a_transpose_5;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 17] = a_transpose_3 * a_transpose_6;

						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 18] = a_transpose_4 * a_transpose_1;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 19] = a_transpose_4 * a_transpose_2;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 20] = a_transpose_4 * a_transpose_3;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 21] = a_transpose_4 * a_transpose_4;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 22] = a_transpose_4 * a_transpose_5;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 23] = a_transpose_4 * a_transpose_6;

						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 24] = a_transpose_5 * a_transpose_1;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 25] = a_transpose_5 * a_transpose_2;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 26] = a_transpose_5 * a_transpose_3;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 27] = a_transpose_5 * a_transpose_4;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 28] = a_transpose_5 * a_transpose_5;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 29] = a_transpose_5 * a_transpose_6;

						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 30] = a_transpose_6 * a_transpose_1;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 31] = a_transpose_6 * a_transpose_2;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 32] = a_transpose_6 * a_transpose_3;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 33] = a_transpose_6 * a_transpose_4;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 34] = a_transpose_6 * a_transpose_5;
						dev_linear_system_left_matrix[x + WIDTH * HEIGHT * 35] = a_transpose_6 * a_transpose_6;
					}
				}
			}
		}
	}
}

void Cholesky_Decomposition(float matrix[][6], int n, float **lower)
{
	// Decomposing a matrix into Lower Triangular
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			float sum = 0.0f;

			if (j == i) // summation for diagnols
			{
				for (int k = 0; k < j; k++)
					sum += pow(lower[j][k], 2);
				lower[j][j] = sqrtf(matrix[j][j] - sum);
			}
			else
			{
				// Evaluating L(i, j) using L(j, j)
				for (int k = 0; k < j; k++)
					sum += (lower[i][k] * lower[j][k]);

				if (isnan(lower[j][j]) || lower[j][j] == 0.0f)
				{
					lower[j][j] = 0.0f;
					lower[i][j] = (matrix[i][j] - sum) / (lower[j][j] + 0.000000001f);
				}
				else
				{
					lower[i][j] = (matrix[i][j] - sum) / (lower[j][j]);
				}
			}
		}
	}
}

void Solve_Linear_System(float *linear_system_left_sum, float *linear_system_right_sum, cv::Mat &refinement_6dof_trans_cv,
						 cv::Mat &global_extrinsic_cv, cv::Mat &updated_6dof_cv, int i)
{
	float linear_system_left_sum_2d[6][6] = {{linear_system_left_sum[0], linear_system_left_sum[1], linear_system_left_sum[2], linear_system_left_sum[3], linear_system_left_sum[4], linear_system_left_sum[5]},
											 {linear_system_left_sum[6], linear_system_left_sum[7], linear_system_left_sum[8], linear_system_left_sum[9], linear_system_left_sum[10], linear_system_left_sum[11]},
											 {linear_system_left_sum[12], linear_system_left_sum[13], linear_system_left_sum[14], linear_system_left_sum[15], linear_system_left_sum[16], linear_system_left_sum[17]},
											 {linear_system_left_sum[18], linear_system_left_sum[19], linear_system_left_sum[20], linear_system_left_sum[21], linear_system_left_sum[22], linear_system_left_sum[23]},
											 {linear_system_left_sum[24], linear_system_left_sum[25], linear_system_left_sum[26], linear_system_left_sum[27], linear_system_left_sum[28], linear_system_left_sum[29]},
											 {linear_system_left_sum[30], linear_system_left_sum[31], linear_system_left_sum[32], linear_system_left_sum[33], linear_system_left_sum[34], linear_system_left_sum[35]}};

	// solve Ax = B, where A = a'a, B = a'b
	// first get a'
	float **lower_triangle_matrix;
	lower_triangle_matrix = new float *[6];
	for (int ii = 0; ii < 6; ii++)
	{
		lower_triangle_matrix[ii] = new float[6];
		std::fill(lower_triangle_matrix[ii], lower_triangle_matrix[ii] + 6, 0.0f);
	}
	Cholesky_Decomposition(linear_system_left_sum_2d, 6, lower_triangle_matrix);

	float *lower_triangle_vector = new float[36];
	for (int x = 0; x < 6; x++)
	{
		for (int y = 0; y < 6; y++)
		{
			lower_triangle_vector[x * 6 + y] = lower_triangle_matrix[x][y];
		}
	}

	// create OpenCV 2D
	cv::Mat lower_triangle_matrix_cv(6, 6, CV_32F);
	cv::Mat upper_triangle_matrix_cv(6, 6, CV_32F);
	cv::Mat linear_system_right_sum_cv(6, 1, CV_32F);

	// get a'
	memcpy(lower_triangle_matrix_cv.data, lower_triangle_vector, 6 * 6 * sizeof(float));
	// get a
	cv::transpose(lower_triangle_matrix_cv, upper_triangle_matrix_cv);

	cv::Mat rightside_vector_cv(6, 1, CV_32F);
	memcpy(linear_system_right_sum_cv.data, linear_system_right_sum, 6 * sizeof(float));
	// solve B = a'b, get b
	cv::solve(lower_triangle_matrix_cv, linear_system_right_sum_cv, rightside_vector_cv, cv::DECOMP_LU);
	// solve ax = b, where x = {Beta, Gamma, Alpha, tx, ty, tz}
	cv::solve(upper_triangle_matrix_cv, rightside_vector_cv, updated_6dof_cv, cv::DECOMP_LU);

	// clean all values for the next run of icp
	delete[] lower_triangle_matrix;
	delete[] lower_triangle_vector;
}

__global__ void Sum_Array(const float *g_idata, float *g_odata, const int k, const int HEIGHT, const int WIDTH)
{
	extern __shared__ float sdata[64];

	// reduction
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i + WIDTH * HEIGHT * k];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

extern "C" void Estimate_Sensor_Pose(const int &HEIGHT, const int &WIDTH,
									 const cv::Mat &cam_intrinsic_cv, cv::Mat &global_extrinsic_cv,
									 const cv::Mat &vertices_x_cv, const cv::Mat &vertices_y_cv, const cv::Mat &vertices_z_cv,
									 const cv::Mat &normals_x_cv, const cv::Mat &normals_y_cv, const cv::Mat &normals_z_cv,
									 const cv::Mat &surface_points_x_cv, const cv::Mat &surface_points_y_cv, const cv::Mat &surface_points_z_cv,
									 const cv::Mat &surface_normals_x_cv, const cv::Mat &surface_normals_y_cv, const cv::Mat &surface_normals_z_cv,
									 const cv::Mat &vertex_mask)
{
	cv::Mat inv_cam_intrinsic_cv = cam_intrinsic_cv.inv();
	cv::Mat inv_global_extrinsic_cv = global_extrinsic_cv.inv();
	cv::Mat refinement_6dof_trans_cv;
	cv::Mat frame_to_frame_trans_cv(4, 4, CV_32F);
	global_extrinsic_cv.copyTo(refinement_6dof_trans_cv);

	// host array
	float *linear_system_right_matrix = new float[WIDTH * HEIGHT * SIZE_6];
	float *linear_system_left_matrix = new float[WIDTH * HEIGHT * SIZE_36];
	std::fill_n(linear_system_right_matrix, WIDTH * HEIGHT * SIZE_6, 0.0f);
	std::fill_n(linear_system_left_matrix, WIDTH * HEIGHT * SIZE_36, 0.0f);

	// gpu data allocation
	float *dev_vertices_x = 0, *dev_vertices_y = 0, *dev_vertices_z = 0;
	float *dev_normals_x = 0, *dev_normals_y = 0, *dev_normals_z = 0;

	float *dev_surface_points_x = 0, *dev_surface_points_y = 0, *dev_surface_points_z = 0;
	float *dev_surface_normals_x = 0, *dev_surface_normals_y = 0, *dev_surface_normals_z = 0;

	float *dev_cam_intrinsic = 0, *dev_inv_cam_intrinsic = 0;
	float *dev_global_extrinsic = 0, *dev_inv_global_extrinsic = 0;
	float *dev_refinement_6dof_trans = 0;
	float *dev_frame_to_frame_trans = 0;
	uint8_t *dev_vertex_mask = 0;

	float *dev_linear_system_right_matrix = 0, *dev_linear_system_left_matrix = 0;

	// allocate GPU buffers for data
	cudaMalloc((void **)&dev_linear_system_right_matrix, WIDTH * HEIGHT * sizeof(float) * SIZE_6);
	cudaMalloc((void **)&dev_linear_system_left_matrix, WIDTH * HEIGHT * sizeof(float) * SIZE_36);

	cudaMalloc((void **)&dev_vertices_x, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_vertices_y, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_vertices_z, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_normals_x, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_normals_y, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_normals_z, WIDTH * HEIGHT * sizeof(float));

	cudaMalloc((void **)&dev_surface_points_x, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_surface_points_y, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_surface_points_z, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_surface_normals_x, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_surface_normals_y, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_surface_normals_z, WIDTH * HEIGHT * sizeof(float));

	cudaMalloc((void **)&dev_cam_intrinsic, 9 * sizeof(float));
	cudaMalloc((void **)&dev_global_extrinsic, 16 * sizeof(float));
	cudaMalloc((void **)&dev_inv_cam_intrinsic, 9 * sizeof(float));
	cudaMalloc((void **)&dev_inv_global_extrinsic, 16 * sizeof(float));
	cudaMalloc((void **)&dev_refinement_6dof_trans, 16 * sizeof(float));
	cudaMalloc((void **)&dev_frame_to_frame_trans, 16 * sizeof(float));
	cudaMalloc((void **)&dev_vertex_mask, WIDTH * HEIGHT * sizeof(uint8_t));

	// copy input from host memory to GPU buffers
	cudaMemcpy(dev_vertices_x, vertices_x_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertices_y, vertices_y_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertices_z, vertices_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_normals_x, normals_x_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_normals_y, normals_y_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_normals_z, normals_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_surface_points_x, surface_points_x_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_surface_points_y, surface_points_y_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_surface_points_z, surface_points_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_surface_normals_x, surface_normals_x_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_surface_normals_y, surface_normals_y_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_surface_normals_z, surface_normals_z_cv.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_cam_intrinsic, cam_intrinsic_cv.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inv_cam_intrinsic, inv_cam_intrinsic_cv.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_global_extrinsic, global_extrinsic_cv.data, 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inv_global_extrinsic, inv_global_extrinsic_cv.data, 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vertex_mask, vertex_mask.data, WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyHostToDevice);

	int threads_per_block = 64;
	int blocks_per_grid = (WIDTH * HEIGHT + threads_per_block - 1) / threads_per_block;

	float *block_cumulative_sum = new float[blocks_per_grid];
	std::fill_n(block_cumulative_sum, blocks_per_grid, 0.0f);
	float *dev_block_cumulative_sum = 0;
	cudaMalloc((void **)&dev_block_cumulative_sum, blocks_per_grid * sizeof(float));
	cudaMemcpy(dev_block_cumulative_sum, block_cumulative_sum, blocks_per_grid * sizeof(float), cudaMemcpyHostToDevice);

	// iterative closest point algorithm
	for (int i = 0; i < 10; i++)
	{
		// update refinement 6dof
		cudaMemcpy(dev_refinement_6dof_trans, refinement_6dof_trans_cv.data, EXTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
		frame_to_frame_trans_cv = inv_global_extrinsic_cv * refinement_6dof_trans_cv;
		cudaMemcpy(dev_frame_to_frame_trans, frame_to_frame_trans_cv.data, EXTRINSIC_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_linear_system_right_matrix, linear_system_right_matrix, WIDTH * HEIGHT * sizeof(float) * SIZE_6, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_linear_system_left_matrix, linear_system_left_matrix, WIDTH * HEIGHT * sizeof(float) * SIZE_36, cudaMemcpyHostToDevice);

		ICP<<<blocks_per_grid, threads_per_block>>>(dev_vertices_x, dev_vertices_y, dev_vertices_z,
													dev_normals_x, dev_normals_y, dev_normals_z,
													dev_surface_points_x, dev_surface_points_y, dev_surface_points_z,
													dev_surface_normals_x, dev_surface_normals_y, dev_surface_normals_z,
													dev_cam_intrinsic, dev_inv_global_extrinsic, dev_refinement_6dof_trans, dev_frame_to_frame_trans,
													dev_linear_system_right_matrix, dev_linear_system_left_matrix, dev_vertex_mask,
													HEIGHT, WIDTH);
		cudaDeviceSynchronize();

		float *linear_system_right_sum = new float[SIZE_6];
		std::fill_n(linear_system_right_sum, SIZE_6, 0.0f);
		float *linear_system_left_sum = new float[SIZE_36];
		std::fill_n(linear_system_left_sum, SIZE_36, 0.0f);

		for (int k = 0; k < SIZE_6; k++)
		{
			Sum_Array<<<blocks_per_grid, threads_per_block>>>(dev_linear_system_right_matrix, dev_block_cumulative_sum, k, HEIGHT, WIDTH);
			cudaDeviceSynchronize();
			// copy output vector from GPU buffer to host memory
			cudaMemcpy(block_cumulative_sum, dev_block_cumulative_sum, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);
			for (int i = 0; i < blocks_per_grid; i++)
			{
				linear_system_right_sum[k] += block_cumulative_sum[i];
			}
			//std::fill_n(block_cumulative_sum, blocks_per_grid, 0.0f);
			//cudaMemcpy(dev_block_cumulative_sum, block_cumulative_sum, blocks_per_grid * sizeof(float), cudaMemcpyHostToDevice);
		}
		for (int k = 0; k < SIZE_36; k++)
		{
			Sum_Array<<<blocks_per_grid, threads_per_block>>>(dev_linear_system_left_matrix, dev_block_cumulative_sum, k, HEIGHT, WIDTH);
			cudaDeviceSynchronize();
			cudaMemcpy(block_cumulative_sum, dev_block_cumulative_sum, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);
			for (int i = 0; i < blocks_per_grid; i++)
			{
				linear_system_left_sum[k] += block_cumulative_sum[i];
			}
			//std::fill_n(block_cumulative_sum, blocks_per_grid, 0.0f);
			//cudaMemcpy(dev_block_cumulative_sum, block_cumulative_sum, blocks_per_grid * sizeof(float), cudaMemcpyHostToDevice);
		}

		// HAL module
		cv::Mat linear_system_left_sum_cv(SIZE_6, SIZE_6, CV_32F);
		memcpy(linear_system_left_sum_cv.data, linear_system_left_sum, SIZE_36 * sizeof(float));

		cv::Mat linear_system_right_sum_cv(SIZE_6, 1, CV_32F);
		memcpy(linear_system_right_sum_cv.data, linear_system_right_sum, SIZE_6 * 1 * sizeof(float));
		// linear_system_right_sum will return output of updated parameters {Beta, Gamma, Alpha, tx, ty, tz}
		cv::Cholesky(linear_system_left_sum, linear_system_left_sum_cv.step, SIZE_6, linear_system_right_sum, linear_system_right_sum_cv.step, 1);

		cv::Mat updated_6dof_cv(SIZE_6, 1, CV_32F);

		// solve linear system step by step
		//Solve_Linear_System(linear_system_left_sum, linear_system_right_sum, refinement_6dof_trans_cv,
		//	                global_extrinsic_cv, updated_6dof_cv, i);

		memcpy(updated_6dof_cv.data, linear_system_right_sum, SIZE_6 * 1 * sizeof(float));

		cv::Mat sixdof_increment_cv(4, 4, CV_32F);
		sixdof_increment_cv.ptr<float>(0)[0] = 1.0f;
		sixdof_increment_cv.ptr<float>(0)[1] = updated_6dof_cv.ptr<float>(2)[0];
		sixdof_increment_cv.ptr<float>(0)[2] = -updated_6dof_cv.ptr<float>(1)[0];
		sixdof_increment_cv.ptr<float>(0)[3] = updated_6dof_cv.ptr<float>(3)[0];

		sixdof_increment_cv.ptr<float>(1)[0] = -updated_6dof_cv.ptr<float>(2)[0];
		sixdof_increment_cv.ptr<float>(1)[1] = 1.0f;
		sixdof_increment_cv.ptr<float>(1)[2] = updated_6dof_cv.ptr<float>(0)[0];
		sixdof_increment_cv.ptr<float>(1)[3] = updated_6dof_cv.ptr<float>(4)[0];

		sixdof_increment_cv.ptr<float>(2)[0] = updated_6dof_cv.ptr<float>(1)[0];
		sixdof_increment_cv.ptr<float>(2)[1] = -updated_6dof_cv.ptr<float>(0)[0];
		sixdof_increment_cv.ptr<float>(2)[2] = 1.0f;
		sixdof_increment_cv.ptr<float>(2)[3] = updated_6dof_cv.ptr<float>(5)[0];

		sixdof_increment_cv.ptr<float>(3)[0] = 0.0f;
		sixdof_increment_cv.ptr<float>(3)[1] = 0.0f;
		sixdof_increment_cv.ptr<float>(3)[2] = 0.0f;
		sixdof_increment_cv.ptr<float>(3)[3] = 1.0f;

		// update transform
		refinement_6dof_trans_cv = sixdof_increment_cv * refinement_6dof_trans_cv;
		global_extrinsic_cv = refinement_6dof_trans_cv;

		// clean all values for the next run of icp
		delete[] linear_system_right_sum;
		delete[] linear_system_left_sum;
	}

	// free device array
	cudaFree(dev_vertices_x);
	cudaFree(dev_vertices_y);
	cudaFree(dev_vertices_z);
	cudaFree(dev_normals_x);
	cudaFree(dev_normals_y);
	cudaFree(dev_normals_z);

	cudaFree(dev_surface_points_x);
	cudaFree(dev_surface_points_y);
	cudaFree(dev_surface_points_z);
	cudaFree(dev_surface_normals_x);
	cudaFree(dev_surface_normals_y);
	cudaFree(dev_surface_normals_z);

	cudaFree(dev_cam_intrinsic);
	cudaFree(dev_global_extrinsic);
	cudaFree(dev_inv_cam_intrinsic);
	cudaFree(dev_inv_global_extrinsic);
	cudaFree(dev_refinement_6dof_trans);
	cudaFree(dev_frame_to_frame_trans);
	cudaFree(dev_vertex_mask);
	cudaFree(dev_linear_system_right_matrix);
	cudaFree(dev_linear_system_left_matrix);
	cudaFree(dev_block_cumulative_sum);

	delete[] block_cumulative_sum;
	delete[] linear_system_right_matrix;
	delete[] linear_system_left_matrix;
}