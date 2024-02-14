#include <opencv2/opencv.hpp>

void bilateral_filter(Mat& image, Mat& filtered_image) {
    // Parameters
    int diameter = 19;
    double sigma_color = 30;
    double sigma_space = 20;
    cout << "[SR7 INFO Filtering] Bilateral filter parameters: diameter=" << diameter << ", sigma_color=" << sigma_color << ", sigma_space=" << sigma_space << endl;

    cout << "[SR7 INFO Filtering] Applying bilateral filter..." << endl;
    // Apply bilateral filter
    cv::bilateralFilter(image, filtered_image, diameter, sigma_color, sigma_space);
    cout << "[SR7 INFO Filtering] Bilateral filter successfully applied!" << endl;
}