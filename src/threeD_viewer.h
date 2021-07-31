#ifndef THREED_VIEWER_H_
#define THREED_VIEWER_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/pcl_visualizer.h>

class ThreeDViewer
{
public:
    ThreeDViewer(int height, int width, std::shared_ptr<Kinfu::KinfuPipeline> kinect_fusion_system);
    ~ThreeDViewer();

    void SetUpPointClouds(std::shared_ptr<Kinfu::KinfuPipeline> &kinect_fusion_system);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> point_cloud;

private:
    int _rows{0};
    int _cols{0};
};

#endif