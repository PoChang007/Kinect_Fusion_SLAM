#include "kinfu_pipeline.h"

int main()
{
    // determine how to read the image sequence
    int start_frame = 20;
    int next_frame = 21;
    int per_nth_frame = 1;
    int end_frame = 70;

    std::unique_ptr<Kinfu::KinfuPipeline> kinectFusionSystem = std::make_unique<Kinfu::KinfuPipeline>(start_frame, next_frame, per_nth_frame, end_frame);
    kinectFusionSystem->StartProcessing();

    // cv::waitKey(50000);

    return 0;
}
