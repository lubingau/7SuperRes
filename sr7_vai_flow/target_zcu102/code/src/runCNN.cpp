#include <opencv2/opencv.hpp>

void interpolateImages(const std::vector<cv::Mat>& inputImages, std::vector<cv::Mat>& outputImages) {
    for (const auto& inputImage : inputImages) {
        cv::Mat outputImage;
        cv::resize(inputImage, outputImage, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
        outputImages.push_back(outputImage);
    }
}

