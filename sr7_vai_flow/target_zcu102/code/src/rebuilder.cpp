#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>


#define IMG_HEIGHT 1419
#define IMG_WIDTH 1672

using namespace cv;
using namespace std;

void ListImages(string const &path, vector<string> &images_list) {
  images_list.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
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

    std::vector<string> images_list;
    ListImages(patch_folder, images_list);
    cout << "Found " << images_list.size() << " patches\n";

    for (const auto& patch_name : images_list) {
        string patch_path = patch_folder + "/" + patch_name;
        Mat patch = imread(patch_path);
        if (patch.empty()) {
            cout << "Error: Unable to load patch from " << patch_path << endl;
            return;
        }

        std::size_t i_pos = patch_name.find("i");
        std::size_t j_pos = patch_name.find("j");
        std::size_t endi_pos = patch_name.find("endi");
        std::size_t endj_pos = patch_name.find("endj");

        int row = stoi(patch_name.substr(i_pos + 1, endi_pos - i_pos - 1));
        int col = stoi(patch_name.substr(j_pos + 1, endj_pos - j_pos - 1));

        Rect patch_rect(col, row, patch.cols, patch.rows);

        image(patch_rect) += patch;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./image_patcher <patch_folder>\n";
        return -1;
    }

    string patch_folder = argv[1];

    Mat image(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);

    rebuild_image(image, patch_folder);

    imwrite("reconstructed_image.png", image);
    return 0;
}