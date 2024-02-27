#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include "rebuilder.cpp"
#include "runCNN_test.cpp"
//#include "runCNN.cpp"

#include "debug.h"

using namespace std;
using namespace cv;

double calculateMSE(const Mat& img1, const Mat& img2) {
    Mat diff;
    absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_64F); // convert to floating point
    Mat squaredDiff = diff.mul(diff);
    Scalar mse = mean(squaredDiff);
    return mse[0]; // return the mean squared error
}

double calculatePSNR(double mse, double maxValue) {
    if (mse == 0.0) {
        return 0; // PSNR is infinite, return 0 in this case
    }
    double psnr = 10.0 * log10((maxValue * maxValue) / mse);
    return psnr;
}

void PSNR(vector<Mat>& predict_patches, vector<Mat>& gt_patches) {
    /* 
    A method to calculate the PSNR between two images.

    Args:
        predict_patches: The blurred image
        gt_patches: The grand truth image
    */

    double totalPSNR = 0.0;
    int numImages = predict_patches.size();

    for (int i = 0; i < numImages; ++i) {
        double mse = calculateMSE(predict_patches[i], gt_patches[i]);
        double psnr = calculatePSNR(mse, 255.0); // assuming pixel values are in [0, 255]
        totalPSNR += psnr;
#if DEBUG_PSNR
        cout << "[SR7 INFO Evaluation] PSNR for image " << i << ": " << psnr << " dB" << endl;
#endif
    }

    double averagePSNR = totalPSNR / numImages;
    cout << "[SR7 INFO Evaluation] Average PSNR: " << averagePSNR << " dB" << endl;
}

int main(int argc, char const *argv[]){
    
    if (argc < 4) {
         cout << "Usage: ./eval <blr_patches_path> <gt_patches_path> <path_xmodel> \n";
         return -1;
    }
    
    string blr_patches_path = argv[1];
    string gt_patches_path = argv[2];
    string path_xmodel = argv[3];

    vector<Mat> img_blr_patch_vec;
    vector<Mat> img_gt_patch_vec;
    vector<Mat> img_predict_patch_vec;
    vector<string> name_blr_vec;
    vector<string> name_gt_vec;


    ListImages(blr_patches_path, name_blr_vec, img_blr_patch_vec);
    ListImages(gt_patches_path, name_gt_vec, img_gt_patch_vec);

    int threads = 6;

#if DEBUG_SAVE
    string debug_folder = "../../debug/";
    string debug_eval_input_folder = debug_folder + "eval/input/";
    string debug_eval_output_folder = debug_folder + "eval/output/";

    /* extrapolateImages(img_blr_patch_vec, img_predict_patch_vec);
    
    cout << "[SR7 INFO Evaluation] Saving input and predicted images..." << endl;

    for (int i = 0; i < img_predict_patch_vec.size(); i++){
        imwrite(debug_eval_input_folder + name_blr_vec[i], img_blr_patch_vec[i]);
        imwrite(debug_eval_output_folder + name_gt_vec[i], img_predict_patch_vec[i]);
    }
    cout << "[SR7 INFO Evaluation] Finished to save input and predicted images!" << endl; */

    runCNN(img_patch_vec_temp, doub_img_patch_vec_temp, path_xmodel, threads, debug_eval_input_folder, debug_eval_output_folder);
#else
    runCNN(img_patch_vec_temp, doub_img_patch_vec_temp, path_xmodel, threads);
    //extrapolateImages(img_patch_vec_temp, doub_img_patch_vec_temp);
#endif

    cout << "[SR7 INFO Evaluation] Calculating PSNR..." << endl;
    PSNR(img_predict_patch_vec, img_gt_patch_vec);
}