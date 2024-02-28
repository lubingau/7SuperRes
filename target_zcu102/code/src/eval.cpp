#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
// #include "runCNN_test.cpp"
#include "runCNN.cpp"

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
#if DEBUG_EVAL_PSNR
        cout << "[SR7 INFO Evaluation] PSNR for image " << i << ": " << psnr << " dB" << endl;
#endif
    }

    double averagePSNR = totalPSNR / numImages;
    cout << "[SR7 INFO Evaluation] Average PSNR: " << averagePSNR << " dB" << endl;
}


void ListImagesDataset(const string &dataset_path, vector<string> &name_vec_blr, vector<Mat>& img_patch_vec_blr, vector<string> &name_vec_gt, vector<Mat>& img_patch_vec_gt) {
    /*
    A method to list all the images in a directory and store them in a vector.

    Args:
        dataset_path: The path of the directory containing the blr and gt images.
        name_vec: The vector where the names of the images will be stored.
        img_patch_vec: The vector where the images will be stored.
    */

    name_vec_blr.clear();
    img_patch_vec_blr.clear();
    name_vec_gt.clear();
    img_patch_vec_gt.clear();

    struct dirent *entry;
    string blr_path = dataset_path + "/blr";
    string gt_path = blr_path + "/../gt";

    /* Check if path is a valid directory path. */
    struct stat s;
    if (stat(blr_path.c_str(), &s) != 0 || !S_ISDIR(s.st_mode)) {
        fprintf(stderr, "[SR7 ERROR Rebuilder - ListImages] %s is not a valid directory!\n", blr_path.c_str());
        exit(1);
    }
    if (stat(gt_path.c_str(), &s) != 0 || !S_ISDIR(s.st_mode)) {
        fprintf(stderr, "[SR7 ERROR Rebuilder - ListImages] %s is not a valid directory!\n", gt_path.c_str());
        exit(1);
    }

    DIR *dir = opendir(blr_path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "[SR7 ERROR Rebuilder - ListImages] Open %s path failed.\n", blr_path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            string name = entry->d_name;
            if (name.find(".png") != string::npos) {
                string blr_name = blr_path + "/" + name;
                // replace blr with gt in name
                size_t pos = name.find("blr");
                string gt_name = name.replace(pos, 3, "gt");
                gt_name = gt_path + "/" + name;
                Mat blr_img = imread(blr_name, IMREAD_COLOR);
                Mat gt_img = imread(gt_name, IMREAD_COLOR);
                if (blr_img.empty()) {
                    fprintf(stderr, "[SR7 ERROR Rebuilder - ListImages] %s is not a valid image!\n", blr_name.c_str());
                    exit(1);
                }
                if (gt_img.empty()) {
                    fprintf(stderr, "[SR7 ERROR Rebuilder - ListImages] %s is not a valid image!\n", gt_name.c_str());
                    exit(1);
                }
                name_vec_blr.push_back(name);
                img_patch_vec_blr.push_back(blr_img);
                name_vec_gt.push_back(name);
                img_patch_vec_gt.push_back(gt_img);
            }
        }
    }

    closedir(dir);
}


int main(int argc, char const *argv[]){
    
    if (argc != 3) {
         cout << "Usage: ./eval <dataset_path> <path_xmodel> \n";
         return -1;
    }
    
    string dataset_path = argv[1];
    string path_xmodel = argv[2];

    vector<Mat> img_blr_patch_vec;
    vector<Mat> img_gt_patch_vec;
    vector<Mat> img_predict_patch_vec;
    vector<string> name_blr_vec;
    vector<string> name_gt_vec;

    ListImagesDataset(dataset_path, name_blr_vec, img_blr_patch_vec, name_gt_vec, img_gt_patch_vec);

    int threads = 6;

#if DEBUG_EVAL_SAVE
    string debug_folder = "/home/petalinux/target_zcu102/debug/";
    string debug_eval_input_folder = debug_folder + "eval/input/";
    string debug_eval_output_folder = debug_folder + "eval/output/";

    /* extrapolateImages(img_blr_patch_vec, img_predict_patch_vec);
    
    cout << "[SR7 INFO Evaluation] Saving input and predicted images..." << endl;

    for (int i = 0; i < img_predict_patch_vec.size(); i++){
        imwrite(debug_eval_input_folder + name_blr_vec[i], img_blr_patch_vec[i]);
        imwrite(debug_eval_output_folder + name_gt_vec[i], img_predict_patch_vec[i]);
    }
    cout << "[SR7 INFO Evaluation] Finished to save input and predicted images!" << endl; */
    cout << debug_eval_input_folder << endl;
    cout << debug_eval_output_folder << endl;
    runCNN(img_blr_patch_vec, img_predict_patch_vec, path_xmodel, threads, debug_eval_input_folder, debug_eval_output_folder);
#else
    // runCNN(img_blr_patch_vec, img_predict_patch_vec, path_xmodel, threads);
    extrapolateImages(img_blr_patch_vec, img_predict_patch_vec);
    string debug_folder = "/home/petalinux/target_zcu102/debug/";
    string debug_eval_input_folder = debug_folder + "eval/input/";
    string debug_eval_output_folder = debug_folder + "eval/output/";
    
    cout << "[SR7 INFO Evaluation] Saving input and predicted images..." << endl;
    for (int i = 0; i < 5; i++){
        imwrite(debug_eval_input_folder + name_gt_vec[i], img_gt_patch_vec[i]);
        imwrite(debug_eval_input_folder + name_blr_vec[i], img_blr_patch_vec[i]);
        imwrite(debug_eval_output_folder + "predict_" + name_gt_vec[i], img_predict_patch_vec[i]);
    }
    cout << "[SR7 INFO Evaluation] Finished to save input and predicted images!" << endl;
#endif

    cout << "[SR7 INFO Evaluation] Calculating PSNR..." << endl;
    PSNR(img_predict_patch_vec, img_gt_patch_vec);
}