#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <cmath>

#include "debug.h"

using namespace cv;
using namespace std;

void argmax(const Mat& matrix, Mat& Mask, const int channel) {
    Point maxLoc;
    double maxVal;

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j+=channel) {
            //extract the channel of the matrix
            Mat channel_matrix = matrix(Rect(j, i, channel, 1));
            // cout << "Channel matrix: " << channel_matrix << endl;
            minMaxLoc(channel_matrix, nullptr, &maxVal, nullptr, &maxLoc);
            // cout << "Max loc x: " << maxLoc.x << " y: " << maxLoc.y << endl;
            Mask.at<uchar>(i, j/channel) = (uchar)maxLoc.x;
        }
    }
}

void mask2mat(int8_t* inputBuffer, int8_t* outputBuffer, const int n_patches, const int patch_size, const int n_classes
#if DEBUG_REBUILDER
    , const string& output_images_folder
#endif
) {
    /*
    A method to rebuild an image from patches and apply division rectangles to correct the overlapping regions due to the stride used.

    Args:
        inputBuffer: The buffer containing the patches.
        outputBuffer: The buffer where the masks will be stored.
        n_patches: The number of patches.
        patch_size: The size of the patches.
    if DEBUG:
        output_images_folder: The folder where the patches divied by rectangles will be saved.
    */

    int inSize = patch_size * patch_size * n_classes;
    int outSize = patch_size * patch_size;
    
    for (int n=0; n<n_patches; n++){

        Mat patch(patch_size, n_classes*patch_size, CV_8UC1, &inputBuffer[n * inSize]);

        // cout << "Patch: " << patch << endl;
        
        Mat Mask = Mat::zeros(patch_size, patch_size, CV_8UC1);

        argmax(patch, Mask, n_classes);

        // cout << "Mask: " << Mask << endl;

        memcpy(&outputBuffer[n * outSize], Mask.data, outSize);

        if (n+1 % 100 == 0){
            cout << "\x1b[A";
            cout << "[SR7 INFO Rebuilder] Patches rebuilded: " << n << endl;
        }  

    }
}

int main(int argc, char const *argv[])
{   
    // if (argc < 5) {
    //     cout << "Usage: ./rebuilder <path_image> <patch_folder> <patch_size> <stride>\n";
    //     return -1;
    // }
    // string path_image = argv[1];
    // string patch_folder = argv[2];
    // int patch_size = stoi(argv[3]);
    // float stride = stof(argv[4]);
    
    // initialize the input buffer
    int patch_size = 4;
    int n_patches = 10;
    int n_classes = 7;
    int8_t* inputBuffer = new int8_t[n_patches * patch_size * patch_size * n_classes];
    int8_t* outputBuffer = new int8_t[n_patches * patch_size * patch_size];

    // fill the input buffer with random values
    for (int i = 0; i < n_patches * patch_size * patch_size * n_classes; i++) {
        inputBuffer[i] = rand() % 255;
    }
    
    mask2mat(inputBuffer, outputBuffer, n_patches, patch_size, n_classes);

    int index = 0;
    // get the patches in the output buffer at index
    Mat Mask(patch_size, patch_size, CV_8UC1, &outputBuffer[index * patch_size * patch_size]);
    Mat Matrix(patch_size, n_classes*patch_size, CV_8UC1, &inputBuffer[index * patch_size * patch_size * n_classes]);

    cout << "Matrix: " << Matrix << endl;
    cout << "Mask: " << Mask << endl;
}