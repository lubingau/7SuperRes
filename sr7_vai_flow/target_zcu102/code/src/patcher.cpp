#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "debug.h"

using namespace cv;
using namespace std;


void patch_image(const Mat& image, vector<Mat>& image_arr, vector<string>& name_arr, const int patch_size, const float stride
#if DEBUG_PATCHER
    , const string& output_folder
#endif
) {
    int stride_pixels = patch_size * stride;
    int overlap = patch_size - stride_pixels;

    cout << "[SR7 INFO Patcher] Started patcher with patch size: " << patch_size << " and stride: " << stride << endl;

    // Regular patches
    for (int i = 0; i < image.rows - patch_size - overlap; i += stride_pixels) {
        for (int j = 0; j < image.cols - patch_size - overlap; j += stride_pixels) {
            Rect patch_rect(j, i, patch_size, patch_size);
            Mat patch = image(patch_rect).clone();

            string name = "i" + to_string(j) + "endi_j" + to_string(i) + "endj";

            image_arr.push_back(patch);
            name_arr.push_back(name);
#if DEBUG_PATCHER            
            string filename = output_folder + name + ".png";
            imwrite(filename, patch);
#endif
            if (image_arr.size() % 100 == 0){
                cout << "\x1b[A";
                cout << "[SR7 INFO Patcher] Patches created: " << image_arr.size() << endl;
            }     
        }
    }
    cout << "[SR7 INFO Patcher] Regular patches created: " << image_arr.size() << endl;

    // Edge patches
    int last_patch_j = image.cols - patch_size;
    int last_patch_i = image.rows - patch_size;

    for (int i = 0; i < image.rows - patch_size - overlap; i += stride_pixels) {
        Rect patch_rect(last_patch_j, i, patch_size, patch_size);
        Mat patch = image(patch_rect).clone();

        string name = "i" + to_string(last_patch_j) + "endi_j" + to_string(i) + "endj";
            
        image_arr.push_back(patch);
        name_arr.push_back(name);   
#if DEBUG_PATCHER            
        string filename = output_folder + name + ".png";
        imwrite(filename, patch);
#endif
    }

    // Edge patches
    for (int j = 0; j < image.cols - patch_size - overlap; j += stride_pixels) {
        Rect patch_rect(j, last_patch_i, patch_size, patch_size);
        Mat patch = image(patch_rect).clone();
        
        string name = "i" + to_string(j) + "endi_j" + to_string(last_patch_i) + "endj";
            
        image_arr.push_back(patch);
        name_arr.push_back(name);
#if DEBUG_PATCHER            
        string filename = output_folder + name + ".png";
        imwrite(filename, patch);
#endif
    }

    // Last patch (bottom right corner)
    Rect patch_rect(last_patch_j, last_patch_i, patch_size, patch_size);
    Mat patch = image(patch_rect).clone();
    
    string name = "i" + to_string(last_patch_j) + "endi_j" + to_string(last_patch_i) + "endj";
            
    image_arr.push_back(patch);
    name_arr.push_back(name);
#if DEBUG_PATCHER            
    string filename = output_folder + name + ".png";
    imwrite(filename, patch);
#endif
    cout << "[SR7 INFO Patcher] Edge patches created" << endl;
    cout << "[SR7 INFO Patcher] Total patches created: " << image_arr.size() << endl;
}