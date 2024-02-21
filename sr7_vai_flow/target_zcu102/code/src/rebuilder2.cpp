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


void ListImages(const string &path, vector<string> &name_vec, vector<Mat>& img_patch_vec) {
  name_vec.clear();
  img_patch_vec.clear();
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
        name_vec.push_back(name);
        string img_path = path + "/" + name;
        Mat img = imread(img_path);
        img_patch_vec.push_back(img);
      }
    }
  }

  closedir(dir);
}

void rebuild_image_2(const vector<Mat>& img_patch_vec, const vector<string>& name_vec, Mat& reconstruced_image) {

    int patch_size = 128;
    int stride = 0.9 * patch_size;
    int underlap = stride;
    int overlap = patch_size - stride;

    int n_patch_i = (reconstruced_image.cols - patch_size) / stride + 1;
    int n_patch_j = (reconstruced_image.rows - patch_size) / stride + 1;

    int edge_overlap_i = reconstruced_image.cols - n_patch_i*stride;
    int edge_overlap_j = reconstruced_image.rows - n_patch_j*stride;

    cout << "Stride : " << stride << endl;
    cout << "Underlap : " << underlap << endl;
    cout << "Overlap : " << overlap << endl;

    for (int n=0; n<name_vec.size(); n++){
        string patch_name = name_vec[n];
        Mat patch = img_patch_vec[n];

        size_t i_pos = patch_name.find("i");
        size_t j_pos = patch_name.find("j");
        size_t endi_pos = patch_name.find("endi");
        size_t endj_pos = patch_name.find("endj");

        int row = stoi(patch_name.substr(i_pos + 1, endi_pos - i_pos - 1));
        int col = stoi(patch_name.substr(j_pos + 1, endj_pos - j_pos - 1));

        cout<<"Row : "<<row<<" Col : "<<col<<"\n";

        if ((row == 0) and (col == 0)) {
            Rect patch_right(underlap, 0, overlap, underlap); 
            Rect patch_bottom(0, underlap, underlap, overlap);
            Rect patch_corner(underlap, underlap, overlap, overlap);

            patch(patch_right) *= 0.5; 
            patch(patch_bottom) *= 0.5; 
            patch(patch_corner) *=0.25;
        }

        else if ((row!=0) and (col==0) and (row<=reconstruced_image.cols-2*patch_size)) {
            Rect patch_left_up(0, 0, overlap, underlap);
            Rect patch_left_bottom(0, underlap, overlap, overlap);
            Rect patch_bottom(overlap, underlap, underlap-1*overlap, overlap);
            Rect patch_right_bottom(underlap, underlap, overlap, overlap);
            Rect patch_right_up(underlap, 0, overlap, underlap);

            patch(patch_left_up) *= 0.5;
            patch(patch_left_bottom) *= 0.25;
            patch(patch_bottom) *= 0.5;
            patch(patch_right_bottom) *= 0.25;
            patch(patch_right_up) *= 0.5;

            //imwrite("/home/eau_kipik/Images/patch_"+to_string(n)+".png",patch);
        }
        else if ((col!=0) and (row==0) and (col<=reconstruced_image.rows-2*patch_size)){
            Rect patch_left_up(0, 0, underlap, overlap);
            Rect patch_right_up(underlap, 0, overlap, overlap);
            Rect patch_right_mid(underlap, overlap, overlap, underlap-1*overlap);
            Rect patch_right_bottom(underlap, underlap, overlap, overlap);
            Rect patch_left_bottom(0, underlap, underlap, overlap);

            patch(patch_left_up) *= 0.5;
            patch(patch_right_up) *= 0.25;
            patch(patch_right_mid) *= 0.5;
            patch(patch_right_bottom) *= 0.25;
            patch(patch_left_bottom) *= 0.5;

            //imwrite("/home/eau_kipik/Images/patch_"+to_string(n)+".png",patch);
        }

        else if ((col!=0) and (row!=0) and (col<=reconstruced_image.rows-2*patch_size) and (row<=reconstruced_image.cols-2*patch_size)){
            Rect patch_corner_left_up(0, 0, overlap, overlap);
            Rect patch_corner_right_up(underlap, 0, overlap, overlap);
            Rect patch_corner_right_bottom(underlap, underlap, overlap, overlap);
            Rect patch_corner_left_bottom(0, underlap, overlap, overlap);

            Rect patch_left(0, overlap, overlap, underlap-1*overlap);
            Rect patch_right(underlap, overlap, overlap, underlap-1*overlap);
            Rect patch_bottom(overlap, underlap, underlap-1*overlap, overlap);
            Rect patch_up(overlap, 0, underlap-1*overlap, overlap);

            patch(patch_corner_left_up) *= 0.25;
            patch(patch_corner_right_up) *= 0.25;
            patch(patch_corner_right_bottom) *= 0.25;
            patch(patch_corner_left_bottom) *= 0.25;

            patch(patch_left) *= 0.5;
            patch(patch_right) *= 0.5;
            patch(patch_bottom) *= 0.5;
            patch(patch_up) *= 0.5;

            //imwrite("/home/eau_kipik/Images/patch_"+to_string(n)+".png",patch);
        }

        else if ((row ==(n_patch_i-1)*stride) and (col == 0)){
            Rect patch_left_up(0, 0, overlap, underlap);
            Rect patch_left_bottom(0, underlap, overlap, overlap);
            Rect patch_bottom_mid(overlap, underlap, patch_size - edge_overlap_i, overlap);
            Rect patch_right_bottom(patch_size-1*edge_overlap_i+1*overlap, underlap, edge_overlap_i-1*overlap, overlap);
            Rect patch_right_up(patch_size-1*edge_overlap_i+1*overlap,0,edge_overlap_i-1*overlap, underlap);

            patch(patch_left_up) *= 0.5;
            patch(patch_left_bottom) *= 0.25;
            patch(patch_bottom_mid) *= 0.5;
            patch(patch_right_bottom) *= 0.25;
            patch(patch_right_up) *= 0.5;

            //imwrite("/home/eau_kipik/Images/patch_"+to_string(n)+".png",patch);
        }

        else if ((row ==(n_patch_i-1)*stride) and (col != 0) and (col<=reconstruced_image.rows-2*patch_size)){
            Rect patch_left_up(0, 0, overlap, overlap);
            Rect patch_left_mid(0, overlap, overlap, underlap-1*overlap);
            Rect patch_left_bottom(0, underlap, overlap, overlap);
            Rect patch_bottom_mid(overlap, underlap, patch_size - edge_overlap_i, overlap);
            Rect patch_up_mid(overlap, 0, patch_size - edge_overlap_i, overlap);
            Rect patch_right_bottom(patch_size-1*edge_overlap_i+1*overlap, underlap, edge_overlap_i-1*overlap, overlap);
            Rect patch_right_up(patch_size-1*edge_overlap_i+1*overlap,0,edge_overlap_i-1*overlap, underlap);

            patch(patch_left_up) *= 0.25;
            patch(patch_left_mid) *= 0.5;
            patch(patch_left_bottom) *= 0.25;
            patch(patch_bottom_mid) *= 0.5;
            patch(patch_up_mid) *= 0.5;
            patch(patch_right_bottom) *= 0.25;
            patch(patch_right_up) *= 0.5;

            //imwrite("/home/eau_kipik/Images/patch_"+to_string(n)+".png",patch);
        }

        else if ((col ==(n_patch_j-1)*stride) and (row != 0) and (row<=reconstruced_image.cols-2*patch_size)){
            Rect patch_left_up(0, 0, overlap, overlap);
            Rect patch_left_mid(0, overlap, overlap, underlap-1*overlap);
            Rect patch_left_bottom(0, underlap, overlap, overlap);
            Rect patch_bottom_mid(overlap, underlap, patch_size - edge_overlap_i, overlap);
            Rect patch_up_mid(overlap, 0, patch_size - edge_overlap_i, overlap);
            Rect patch_right_bottom(patch_size-1*edge_overlap_i+1*overlap, underlap, edge_overlap_i-1*overlap, overlap);
            Rect patch_right_up(patch_size-1*edge_overlap_i+1*overlap,0,edge_overlap_i-1*overlap, underlap);

            patch(patch_left_up) *= 0.25;
            patch(patch_left_mid) *= 0.5;
            patch(patch_left_bottom) *= 0.25;
            patch(patch_bottom_mid) *= 0.5;
            patch(patch_up_mid) *= 0.5;
            patch(patch_right_bottom) *= 0.25;
            patch(patch_right_up) *= 0.5;

            imwrite("/home/eau_kipik/Images/patch_"+to_string(n)+".png",patch);
        }
            
        Rect patch_rect(row, col, patch.rows, patch.cols);
        reconstruced_image(patch_rect) += patch;
    }

}

int main(int argc, char const *argv[])
{   int IMG_WIDTH = 2455/2;
    int IMG_HEIGHT = 2118/2;
    string patch_folder = "/home/eau_kipik/Images/patcher/";
    vector<Mat> img_patch_vec;
    vector<string> name_vec;
    ListImages(patch_folder, name_vec, img_patch_vec);
    Mat reconstructed_image(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);
    
    cout << "[SR7 INFO Rebuilder] Found " << name_vec.size() << " patches\n";
    rebuild_image_2(img_patch_vec, name_vec, reconstructed_image);
    
    imwrite("/home/eau_kipik/Images/reconstructed_image.png", reconstructed_image);
    return 0;
}
