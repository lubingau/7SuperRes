#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

#include "debug.h"

using namespace cv;
using namespace std;


void rebuild_image_with_mask(const vector<Mat>& img_patch_vec, const vector<string>& name_vec, Mat& reconstructed_image
#if DEBUG_REBUILDER
    , const string& output_images_folder
#endif
) {
    /*
    A method to rebuild an image from patches and apply a mask to correct the overlapping regions due to the stride used.

    Args:
        img_patch_vec: The vector containing the patches to be used to rebuild the image.
        name_vec: The vector containing the names of the patches.
        reconstructed_image: The output image after rebuilding.
    if DEBUG:
        output_images_folder: The folder where the mask and the sum image will be saved.
    */

    cout << "[SR7 INFO Rebuilder] Start to rebuild image" << endl;
    cout << "[SR7 INFO Rebuilder] Found " << img_patch_vec.size() << " patches\n" << endl;

    Mat mask_patch(img_patch_vec[0].size(), CV_8U, Scalar(1));
    Mat mask(reconstructed_image.size(), CV_8U, Scalar(0));
    Mat sum_image(reconstructed_image.size(), CV_16UC3, Scalar(0, 0, 0));

    for (int n=0; n<img_patch_vec.size(); n++) {
        Mat patch = img_patch_vec[n];
        string patch_name = name_vec[n];

        size_t i_pos = patch_name.find("i");
        size_t j_pos = patch_name.find("j");
        size_t endi_pos = patch_name.find("endi");
        size_t endj_pos = patch_name.find("endj");

        int row = stoi(patch_name.substr(i_pos + 1, endi_pos - i_pos - 1));
        int col = stoi(patch_name.substr(j_pos + 1, endj_pos - j_pos - 1));

        Rect patch_rect(2*row, 2*col, patch.rows, patch.cols);

        sum_image(patch_rect) += patch;
        mask(patch_rect) += mask_patch;

        if ((n-1 % 100 == 0)or(n==img_patch_vec.size()-1)) {
            cout << "\x1b[A";
            cout << "[SR7 INFO Rebuilder] " << n+1 << " patches added" << endl;
        }
    }
    cout << "[SR7 INFO Rebuilder] Total of " << img_patch_vec.size() << " patches added" << endl;

    cout << "[SR7 INFO Rebuilder] Applying mask" << endl;
    // Iterate over each pixel
    for (int x = 0; x < reconstructed_image.rows; ++x) {
        for (int y = 0; y < reconstructed_image.cols; ++y) {
            int mask_pixel = mask.at<uchar>(x, y);
            if (mask_pixel != 1) {
                Vec3w image_pixel = sum_image.at<Vec3w>(x, y);
                reconstructed_image.at<Vec3w>(x, y) = (Vec3w)(image_pixel / mask_pixel);
            }
            else {
                reconstructed_image.at<Vec3w>(x, y) = sum_image.at<Vec3w>(x, y);
            }
        }
    }
#if DEBUG_REBUILDER
    // multiply the mask by 32 to make it visible
    mask *= 32;
    string mask_filename = output_images_folder + "mask.png";
    string sum_image_filename = output_images_folder + "sum_image.png";
    sum_image.convertTo(sum_image, CV_8UC3);
    imwrite(mask_filename, mask);
    imwrite(sum_image_filename, sum_image);
    mask.release();
    sum_image.release();
#else
    mask.release();
    sum_image.release();
#endif
    cout << "[SR7 INFO Rebuilder] Mask applied!" << endl;
}