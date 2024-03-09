#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "debug.h"

using namespace cv;
using namespace std;


void patch_image(const Mat& image, int8_t* inputBuffer, int16_t* posBuffer, const int patch_size, const float stride, const int bufferSize
#if DEBUG_PATCHER
    , const string& output_folder
#endif
) {
    /*
    A method to create patches from an input image.

    Args:
        image: The input image to be patched.
        image_arr: The vector where the patches will be stored.
        name_arr: The vector where the names of the patches will be stored.
        patch_size: The size of the patches.
        stride: The stride used to create the patches.
    If DEBUG:
        output_folder: The folder where the patches will be saved.
    */

    int stride_pixels = patch_size * stride;
    int overlap = patch_size - stride_pixels;
    int channels = image.channels();
    int inSize = patch_size * patch_size * channels;
    int posSize = sizeof(u_int16_t);

    cout << "[SR7 INFO Patcher] Started patcher with patch size: " << patch_size << " and stride: " << stride << endl;

    // Regular patches
    int n_patches = 0;
    for (int i = 0; i < image.rows - patch_size - overlap; i += stride_pixels) {
        for (int j = 0; j < image.cols - patch_size - overlap; j += stride_pixels) {
            Rect patch_rect(j, i, patch_size, patch_size);
            Mat patch = image(patch_rect).clone()/4; // Divide by 4 to scaling (input_scale/255 = 64/255 = 1/4)

            memcpy(&inputBuffer[n_patches * inSize], patch.data, inSize);
            memcpy(&posBuffer[2*n_patches], &i, posSize);
            memcpy(&posBuffer[2*n_patches + 1], &j, posSize);

#if DEBUG_PATCHER            
            string filename = output_folder + to_string(i) + "_" + to_string(j) + ".png";
            Mat debug(patch_size, patch_size, CV_8UC3, &inputBuffer[n_patches * inSize]);
            imwrite(filename, debug);
#endif
            if (n_patches+1 % 100 == 0){
                cout << "\x1b[A";
                cout << "[SR7 INFO Patcher] Patches created: " << n_patches << endl;
            }  
            n_patches++;   
        }
    }
    cout << "[SR7 INFO Patcher] Regular patches created: " << n_patches << endl;

    // Edge patches
    int last_patch_i = image.rows - patch_size;
    int last_patch_j = image.cols - patch_size;

    for (int i = 0; i < image.rows - patch_size - overlap; i += stride_pixels) {
        Rect patch_rect(last_patch_j, i, patch_size, patch_size);
        Mat patch = image(patch_rect).clone()/4;

        memcpy(&inputBuffer[n_patches * inSize], patch.data, inSize);
        memcpy(&posBuffer[2*n_patches], &i, posSize);
        memcpy(&posBuffer[2*n_patches + 1], &last_patch_j, posSize);
        n_patches++;

#if DEBUG_PATCHER            
        string filename = output_folder + to_string(i) + "_" + to_string(last_patch_j) + ".png";
        Mat debug(patch_size, patch_size, CV_8UC3, &inputBuffer[n_patches * inSize]);
        imwrite(filename, debug);
#endif
    }

    // Edge patches
    for (int j = 0; j < image.cols - patch_size - overlap; j += stride_pixels) {
        Rect patch_rect(j, last_patch_i, patch_size, patch_size);
        Mat patch = image(patch_rect).clone()/4;
        
        memcpy(&inputBuffer[n_patches * inSize], patch.data, inSize);
        memcpy(&posBuffer[2*n_patches], &last_patch_i, posSize);
        memcpy(&posBuffer[2*n_patches + 1], &j, posSize);
        n_patches++;

#if DEBUG_PATCHER            
        string filename = output_folder + to_string(last_patch_i) + "_" + to_string(j) + ".png";
        Mat debug(patch_size, patch_size, CV_8UC3, &inputBuffer[n_patches * inSize]);
        imwrite(filename, debug);
#endif
    }

    // Last patch (bottom right corner)
    Rect patch_rect(last_patch_j, last_patch_i, patch_size, patch_size);
    Mat patch = image(patch_rect).clone()/4;
    
    memcpy(&inputBuffer[n_patches * inSize], patch.data, inSize);
    memcpy(&posBuffer[2*n_patches], &last_patch_i, posSize);
    memcpy(&posBuffer[2*n_patches + 1], &last_patch_j, posSize);
    n_patches++;

#if DEBUG_PATCHER            
    string filename = output_folder + to_string(last_patch_i) + "_" + to_string(last_patch_j) + ".png";
    Mat debug(patch_size, patch_size, CV_8UC3, &inputBuffer[n_patches * inSize]);
    imwrite(filename, debug);
#endif
    cout << "[SR7 INFO Patcher] Edge patches created" << endl;
    cout << "[SR7 INFO Patcher] Total patches created: " << n_patches << endl;

    if (bufferSize < n_patches) {
        cerr << "[SR7 ERROR Patcher] Buffer size too small for all patches" << endl;
    }
    else if (bufferSize > n_patches) {
        cout << "[SR7 WARNING Patcher] Buffer size too large for all patches" << endl;
        cout << "[SR7 INFO Patcher] All patches created. Total patches: " << n_patches << endl;
    }
    else {
        cout << "[SR7 INFO Patcher] All patches created. Total patches: " << n_patches << endl;
    }
}