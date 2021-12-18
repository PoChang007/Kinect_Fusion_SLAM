#ifndef THREED_VIEWER_H_
#define THREED_VIEWER_H_

#include <pcl/visualization/pcl_visualizer.h>
#include "kinfu_pipeline.h"

namespace Kinfu
{
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
}

#endif