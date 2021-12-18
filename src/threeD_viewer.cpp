#include "threeD_viewer.h"

namespace Kinfu
{
	ThreeDViewer::ThreeDViewer(int height, int width, std::shared_ptr<Kinfu::KinfuPipeline> kinect_fusion_system)
	{
		_rows = height;
		_cols = width;
		point_cloud.emplace_back(new pcl::PointCloud<pcl::PointXYZRGB>);
		point_cloud[0]->width = (uint32_t)height;
		point_cloud[0]->height = (uint32_t)width;
		point_cloud[0]->is_dense = false;
		point_cloud[0]->points.resize(point_cloud[0]->width * point_cloud[0]->height);
	}

	ThreeDViewer::~ThreeDViewer()
	{
	}

	void ThreeDViewer::SetUpPointClouds(std::shared_ptr<Kinfu::KinfuPipeline> &kinect_fusion_system)
	{
		cv::Mat _inverseColorProjection = kinect_fusion_system->intrinsic_matrix.inv();
		for (int u = 0; u < _rows; u++)
		{
			for (int v = 0; v < _cols; v++)
			{
				int index = u * _cols + v;
				if (kinect_fusion_system->system_utility->ray_casting_data->surface_prediction_z.ptr<float>(u)[v] != 0.f)
				{
					point_cloud[0]->points[index].x = kinect_fusion_system->system_utility->ray_casting_data->surface_prediction_x.ptr<float>(u)[v];
					point_cloud[0]->points[index].y = kinect_fusion_system->system_utility->ray_casting_data->surface_prediction_y.ptr<float>(u)[v];
					point_cloud[0]->points[index].z = kinect_fusion_system->system_utility->ray_casting_data->surface_prediction_z.ptr<float>(u)[v];

					std::uint8_t b((int)kinect_fusion_system->system_utility->color_image.ptr<uchar>(u)[3 * v]);
					std::uint8_t g((int)kinect_fusion_system->system_utility->color_image.ptr<uchar>(u)[3 * v + 1]);
					std::uint8_t r((int)kinect_fusion_system->system_utility->color_image.ptr<uchar>(u)[3 * v + 2]);
					std::uint32_t rgb = (static_cast<std::uint32_t>(r) << 16 |
										 static_cast<std::uint32_t>(g) << 8 | static_cast<std::uint32_t>(b));
					point_cloud[0]->points[index].rgb = *reinterpret_cast<float *>(&rgb);
				}
				else
				{
					point_cloud[0]->points[index].x = 0;
					point_cloud[0]->points[index].y = 0;
					point_cloud[0]->points[index].z = 0;

					std::uint8_t r(0), g(0), b(0);
					std::uint32_t rgb = (static_cast<std::uint32_t>(r) << 16 |
										 static_cast<std::uint32_t>(g) << 8 | static_cast<std::uint32_t>(b));
					point_cloud[0]->points[index].rgb = *reinterpret_cast<float *>(&rgb);
				}
			}
		}
	}
}
