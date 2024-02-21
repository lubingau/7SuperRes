#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

#include "debug.h"

void extrapolateImages(const std::vector<cv::Mat>& inputImages, std::vector<cv::Mat>& outputImages) {
    cout << "[SR7 INFO RunCNN] Interpolating images..." << endl;
    for (const auto& inputImage : inputImages) {
        cv::Mat outputImage;
        cv::resize(inputImage, outputImage, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
        outputImages.push_back(outputImage);
    }
    cout << "[SR7 INFO RunCNN] Interpolation done!" << endl;
}
