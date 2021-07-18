#include "kinfu_pipeline.h"

int main()
{
    std::unique_ptr<Kinfu::KinfuPipeline> kinectFusionSystem = std::make_unique<Kinfu::KinfuPipeline>();
    kinectFusionSystem->StartProcessing();

    cv::waitKey(50000);

    return 0;
}
