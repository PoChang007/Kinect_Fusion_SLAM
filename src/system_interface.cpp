#include "kinfu_pipeline.h"
#include "threeDViewer.h"

int main()
{
    // image size
    int height = 480;
    int width = 640;

    // frame settings
    int start_frame = 20;
    int end_frame = 100;
    int current_frame = start_frame;
    int per_nth_frame = 3;
    int per_nth_render_frame = 5;
    bool firstFrameRender = false;

    std::shared_ptr<Kinfu::KinfuPipeline> kinectFusionSystem = std::make_shared<Kinfu::KinfuPipeline>(height, width);
    std::unique_ptr<ThreeDViewer> pclRender = std::make_unique<ThreeDViewer>(kinectFusionSystem->get_shared_this());

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->initCameraParameters();
    // viewer->registerKeyboardCallback(KeyboardEvent);
    viewer->setCameraPosition(0, 0, -8, 0, 0, 0.1, 0, -1, 0);

    while (!viewer->wasStopped())
    {
        if (!firstFrameRender)
        {
            kinectFusionSystem->InitialProcessing(start_frame);

            pclRender->SetUpPointClouds(kinectFusionSystem);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pclRender->point_cloud[0]);
            viewer->addPointCloud<pcl::PointXYZRGB>(pclRender->point_cloud[0], rgb, std::to_string(current_frame));

            firstFrameRender = true;
            current_frame += 1;
        }
        else if (current_frame <= end_frame)
        {
            kinectFusionSystem->IncomingFrameProcessing(current_frame, per_nth_frame);
            if (current_frame % per_nth_render_frame == 0)
            {
                pclRender->SetUpPointClouds(kinectFusionSystem);
                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pclRender->point_cloud[0]);
                viewer->addPointCloud<pcl::PointXYZRGB>(pclRender->point_cloud[0], rgb, std::to_string(current_frame));
            }
            current_frame += per_nth_frame;
        }
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}
