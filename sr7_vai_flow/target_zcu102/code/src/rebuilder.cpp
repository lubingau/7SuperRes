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


void ListImages(const string &path, vector<string> &images_list) {
  images_list.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "[SR7 ERROR Rebuilder - ListImages] %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "[SR7 ERROR Rebuilder - ListImages] Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
        images_list.push_back(name);
      }
    }
  }

  closedir(dir);
}

void rebuild_image(Mat& image, const string& patch_folder) {

    vector<string> images_list;
    ListImages(patch_folder, images_list);
    cout << "[SR7 INFO Rebuilder] Found " << images_list.size() << " patches\n";

    for (const auto& patch_name : images_list) {
        string patch_path = patch_folder + "/" + patch_name;
        Mat patch = imread(patch_path);
        if (patch.empty()) {
            cout << "[SR7 ERROR Rebuilder] Unable to load patch from " << patch_path << endl;
            return;
        }

        size_t i_pos = patch_name.find("i");
        size_t j_pos = patch_name.find("j");
        size_t endi_pos = patch_name.find("endi");
        size_t endj_pos = patch_name.find("endj");

        int row = stoi(patch_name.substr(i_pos + 1, endi_pos - i_pos - 1));
        int col = stoi(patch_name.substr(j_pos + 1, endj_pos - j_pos - 1));

        Rect patch_rect(2*row, 2*col, patch.rows, patch.cols); // 2*row and 2*col because the image has 2x the size of input sensor image

        image(patch_rect) += patch;
    }
}


void rebuild_image_and_mask(const vector<Mat>& img_patch_vec, const vector<string>& name_vec, Mat& reconstructed_image
#if DEBUG_REBUILDER
    , const string& output_images_folder
#endif
) {
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