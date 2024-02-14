#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

using namespace cv;
using namespace std;

void ListImages(string const &path, vector<string> &images_list) {
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

void rebuild_image(Mat& image, vector<Mat>& img_patch_vec, vector<string>& name_vec) {

    for (int i=0; i<img_patch_vec.size(); i++) {
        Mat patch = img_patch_vec[i];
        string patch_name = name_vec[i];

        size_t i_pos = patch_name.find("i");
        size_t j_pos = patch_name.find("j");
        size_t endi_pos = patch_name.find("endi");
        size_t endj_pos = patch_name.find("endj");

        int row = stoi(patch_name.substr(i_pos + 1, endi_pos - i_pos - 1));
        int col = stoi(patch_name.substr(j_pos + 1, endj_pos - j_pos - 1));

        Rect patch_rect(2*row, 2*col, patch.rows, patch.cols);

        image(patch_rect) += patch;
    }
}

void apply_mask(Mat& image, Mat& mask, Mat& reconstructed_image) {
    cout << image.size() << "  " << mask.size() << endl;
    // Check if the image and matrix have the same size
    if (image.size() != mask.size()) {
        cerr << "[SR7 ERROR Rebuilder] Image and mask must have the same size." << endl;
        return;
    }

    cout << "[SR7 INFO Rebuilder] Applying mask..." << endl;
    // Iterate over each pixel
    for (int x = 0; x < image.rows; ++x) {
        for (int y = 0; y < image.cols; ++y) {
            // Get the pixel values from image and matrix
            Vec3w image_pixel = image.at<Vec3w>(x, y);

            uchar mask_pixel = mask.at<uchar>(x, 3*y); // 3*y because the matrix has only 1 channel

            reconstructed_image.at<Vec3b>(x, y) = (Vec3b)(image_pixel / mask_pixel);
        }
    }
    cout << "[SR7 INFO Rebuilder] Mask applied!" << endl;
}

// int main(int argc, char** argv) {
//     if (argc != 3) {
//         cout << "Usage: ./image_patcher <patch_folder> <path_matrix>\n";
//         return -1;
//     }

//     string patch_folder = argv[1];
//     string path_matrix = argv[2];

    //    Mat matrix;
    //    matrix = imread(path_matrix);
    //    matrix.convertTo(matrix, CV_8U);
    //    cout << "Input matrix shape: "<< matrix.size << endl;

    //    int IMG_HEIGHT = matrix.rows;
    //    int IMG_WIDTH = matrix.cols;

    //    Mat image(IMG_HEIGHT, IMG_WIDTH, CV_16UC3);

//     rebuild_image(image, patch_folder);

//     Mat reconstructed_image(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);


    //    cout << "Reconstructed matrix shape: "<< reconstructed_image.size << endl;


//     mean_matrix(image, matrix, reconstructed_image);

//     // convert image into 8-bit
//     image.convertTo(image, CV_8UC3);
//     imwrite("matrix.png", matrix);
//     imwrite("original_image.png", image);
//     imwrite("reconstructed_image.png", reconstructed_image);
//     return 0;
// }