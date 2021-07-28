#include "kinfu_pipeline.h"

__global__ void Bilateral_Filtering(const float *dev_depth_Image_array, float *dev_bilateral_output_array,
									const float *dev_spatial_kernel_array, const float *dev_depth_Image_index_y, const float *dev_depth_Image_index_x,
									const int bw_radius, const float sigma_r,
									const int HEIGHT, const int WIDTH)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= WIDTH * HEIGHT)
		return;

	if ((dev_depth_Image_index_y[x] >= bw_radius && dev_depth_Image_index_y[x] <= (HEIGHT - bw_radius)) &&
		(dev_depth_Image_index_x[x] >= bw_radius && dev_depth_Image_index_x[x] <= (WIDTH - bw_radius)))
	{
		int start_y = dev_depth_Image_index_y[x];
		int start_x = dev_depth_Image_index_x[x];
		float depth_value = dev_depth_Image_array[x];
		const int kernel_length = 2 * bw_radius + 1;
		float *neighboring_pixels = new float[kernel_length * kernel_length];

		float sum = 0.f;
		for (int y = (start_y - bw_radius); y <= start_y + bw_radius; y++)
		{
			for (int x = (start_x - bw_radius); x <= start_x + bw_radius; x++)
			{
				float temp = dev_depth_Image_array[y * WIDTH + x] - depth_value;
				temp = expf(-(temp * temp) / (2 * sigma_r * sigma_r));
				float weights = dev_spatial_kernel_array[((y - start_y) + bw_radius) * kernel_length + ((x - start_x) + bw_radius)] * temp + 0.000000001f;
				neighboring_pixels[((y - start_y) + bw_radius) * kernel_length + ((x - start_x) + bw_radius)] = weights;
				sum = sum + weights;
			}
		}

		float final_sum = 0.f;
		for (int y = (start_y - bw_radius); y <= start_y + bw_radius; y++)
		{
			for (int x = (start_x - bw_radius); x <= start_x + bw_radius; x++)
			{
				float weight_reuslt = neighboring_pixels[((y - start_y) + bw_radius) * kernel_length + ((x - start_x) + bw_radius)] / sum;
				float temp = weight_reuslt * dev_depth_Image_array[y * WIDTH + x];
				final_sum = final_sum + temp;
			}
		}
		dev_bilateral_output_array[x] = final_sum;
		delete[] neighboring_pixels;
	}
}

extern "C" void Bilateral_Filtering(const int &HEIGHT, const int &WIDTH,
									const cv::Mat &depth_Image, cv::Mat &bilateral_output,
									const cv::Mat &spatial_kernel, const float *depth_Image_index_y, const float *depth_Image_index_x,
									const int &bw_radius, const float &sigma_r)
{
	//cudaEvent_t start, stop;
	//float elapsedTime;
	//cudaEventCreate(&start);
	//cudaEventRecord(start, 0);

	int spatial_kernel_size = spatial_kernel.rows;

	// gpu data allocation
	float *dev_depth_Image_array = 0;
	float *dev_bilateral_output_array = 0;
	float *dev_spatial_kernel_array = 0;
	float *dev_depth_Image_index_y = 0;
	float *dev_depth_Image_index_x = 0;

	// allocate gpu buffers for data
	cudaMalloc((void **)&dev_depth_Image_array, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_bilateral_output_array, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_spatial_kernel_array, spatial_kernel_size * spatial_kernel_size * sizeof(float));
	cudaMalloc((void **)&dev_depth_Image_index_y, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void **)&dev_depth_Image_index_x, WIDTH * HEIGHT * sizeof(float));

	// Copy input from host memory to GPU buffers.
	cudaMemcpy(dev_depth_Image_array, depth_Image.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bilateral_output_array, bilateral_output.data, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_spatial_kernel_array, spatial_kernel.data, spatial_kernel_size * spatial_kernel_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_depth_Image_index_y, depth_Image_index_y, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_depth_Image_index_x, depth_Image_index_x, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

	int threadsPerBlock = 64;
	int blocksPerGrid = (WIDTH * HEIGHT + threadsPerBlock - 1) / threadsPerBlock;

	// add kernel here
	Bilateral_Filtering<<<blocksPerGrid, threadsPerBlock>>>(dev_depth_Image_array, dev_bilateral_output_array,
															dev_spatial_kernel_array, dev_depth_Image_index_y, dev_depth_Image_index_x,
															bw_radius, sigma_r,
															HEIGHT, WIDTH);
	cudaDeviceSynchronize();

	// copy output vector from GPU buffer to host memory.
	//cudaMemcpy(bilateral_output.data, dev_bilateral_output_array, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(depth_Image.data, dev_bilateral_output_array, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

	// free Device array
	cudaFree(dev_depth_Image_array);
	cudaFree(dev_bilateral_output_array);
	cudaFree(dev_spatial_kernel_array);
	cudaFree(dev_depth_Image_index_y);
	cudaFree(dev_depth_Image_index_x);

	//cudaEventCreate(&stop);
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&elapsedTime, start, stop);
	//printf("Cuda bilateral filter elapsed time : %f ms\n", elapsedTime);
}