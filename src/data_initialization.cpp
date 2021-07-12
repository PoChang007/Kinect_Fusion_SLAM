#include "kinect_fusion.h"

void initialize_voxel_grid(float* voxel_grid_x, float* voxel_grid_y, float* voxel_grid_z, 
	                       int voxel_grid_x_start, int voxel_grid_y_start, int voxel_grid_z_start, 
	                       int voxel_length, int voxel_width, int voxel_height, float voxel_distance)
{
	int x_index_from_zero = -voxel_grid_x_start;
	int y_index_from_zero = -voxel_grid_y_start;
	int z_index_from_zero = -voxel_grid_z_start;

	for (int x = voxel_grid_x_start; x < voxel_grid_x_start + voxel_length; x++)
	{
		for (int y = voxel_grid_y_start; y < voxel_grid_y_start + voxel_width; y++)
		{
			for (int k = voxel_grid_z_start; k < voxel_grid_z_start + voxel_height; k++)
			{
				voxel_grid_x[(x + x_index_from_zero) * voxel_width * voxel_height + (y + y_index_from_zero) * voxel_height + (k + z_index_from_zero)] = x * voxel_distance + voxel_distance/2.f;
				voxel_grid_y[(x + x_index_from_zero) * voxel_width * voxel_height + (y + y_index_from_zero) * voxel_height + (k + z_index_from_zero)] = y * voxel_distance + voxel_distance/2.f;
				voxel_grid_z[(x + x_index_from_zero) * voxel_width * voxel_height + (y + y_index_from_zero) * voxel_height + (k + z_index_from_zero)] = k * voxel_distance + voxel_distance/2.f;
			}
		}
	}
}

void initialize_int_and_ext_matrix(Mat& intrinsic_matrix, Mat& extrinsic_matrix)
{
	/*Assign intrinsic parameters*/
	intrinsic_matrix.ptr<float>(0)[0] = 524.639f;
	intrinsic_matrix.ptr<float>(0)[1] = 0.0f;
	intrinsic_matrix.ptr<float>(0)[2] = 316.625f;

	intrinsic_matrix.ptr<float>(1)[0] = 0.0f;
	intrinsic_matrix.ptr<float>(1)[1] = 523.503f;
	intrinsic_matrix.ptr<float>(1)[2] = 256.2318f;

	intrinsic_matrix.ptr<float>(2)[0] = 0.0f;
	intrinsic_matrix.ptr<float>(2)[1] = 0.0f;
	intrinsic_matrix.ptr<float>(2)[2] = 1.0f;

	/*Assign extrinsic parameters*/
	extrinsic_matrix.ptr<float>(0)[0] = 1.0f;
	extrinsic_matrix.ptr<float>(0)[1] = 0.0f;
	extrinsic_matrix.ptr<float>(0)[2] = 0.0f;
	extrinsic_matrix.ptr<float>(0)[3] = 0.0f;

	extrinsic_matrix.ptr<float>(1)[0] = 0.0f;
	extrinsic_matrix.ptr<float>(1)[1] = 1.0f;
	extrinsic_matrix.ptr<float>(1)[2] = 0.0f;
	extrinsic_matrix.ptr<float>(1)[3] = 0.0f;

	extrinsic_matrix.ptr<float>(2)[0] = 0.0f;
	extrinsic_matrix.ptr<float>(2)[1] = 0.0f;
	extrinsic_matrix.ptr<float>(2)[2] = 1.0f;
	extrinsic_matrix.ptr<float>(2)[3] = 0.0f;

	extrinsic_matrix.ptr<float>(3)[0] = 0.0f;
	extrinsic_matrix.ptr<float>(3)[1] = 0.0f;
	extrinsic_matrix.ptr<float>(3)[2] = 0.0f;
	extrinsic_matrix.ptr<float>(3)[3] = 1.0f;
}

void load_depth_data(Mat& depth_image, int frame_index)
{
	char filename[200];
	sprintf(filename, "../data/dep%04d.dat", frame_index);
	FILE *fp = fopen(filename, "rb");
	fread(depth_image.data, 2, HEIGHT * WIDTH, fp);
	fclose(fp);

	// convert to float type
	depth_image.convertTo(depth_image, CV_32F);
}

void get_in_range_depth(Mat& depth_image)
{
	// filter out uninterested regions
	float MaxRange = 2000.f;
	float MinRange = 300.f;
	Mat FilterImage;
	// Split into 255 and 0
	cv::inRange(depth_image, MinRange, MaxRange, FilterImage); // get either 255 or 0
	FilterImage = FilterImage / 255;
	FilterImage.convertTo(FilterImage, CV_32F);
	depth_image = depth_image.mul(FilterImage);
}

void initialize_index_matrix(float* depth_Image_index_y, float* depth_Image_index_x)
{
	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			depth_Image_index_y[y * WIDTH + x] = (float)y;
			depth_Image_index_x[y * WIDTH + x] = (float)x;
		}
	}
}

void meshgrid(const cv::Mat& xgv, const cv::Mat& ygv,
			  cv::Mat& X, cv::Mat& Y)
{
	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv, 1, xgv.total(), Y);
}

void create_spatial_kernel(const cv::Range& xgv, const cv::Range& ygv,
	                       cv::Mat& X, cv::Mat& Y)
{
	std::vector<float> t_x, t_y;
	for (float i = xgv.start; i <= xgv.end; i++) {
		t_x.push_back(i);
	}
	for (float i = ygv.start; i <= ygv.end; i++) {
		t_y.push_back(i);
	}
	meshgrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
}

void gaussian_distance_weight(Mat& X, Mat& Y, Mat& weight_d, const float sigma_d)
{
	cv::Mat temp;
	temp = -(X.mul(X, 1) + Y.mul(Y, 1)) / (2 * sigma_d * sigma_d);
	cv::exp(temp, weight_d);
}