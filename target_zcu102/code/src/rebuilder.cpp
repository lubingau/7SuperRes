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


void ListImages(const string &path, vector<string> &name_vec, vector<Mat>& img_patch_vec) {
    /*
    A method to list all the images in a directory and store them in a vector.

    Args:
        path: The path of the directory containing the images.
        name_vec: The vector where the names of the images will be stored.
        img_patch_vec: The vector where the images will be stored.
    */

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

void rebuild_image(const vector<Mat>& img_patch_vec, const vector<string>& name_vec, Mat& reconstruced_image, int patch_size, float stride
#if DEBUG_REBUILDER
    , const string& output_images_folder
#endif
) {
    /*
    A method to rebuild an image from patches and apply division rectangles to correct the overlapping regions due to the stride used.

    Args:
        img_patch_vec: The vector containing the patches to be used to rebuild the image.
        name_vec: The vector containing the names of the patches.
        reconstruced_image: The output image after rebuilding.
        patch_size: The size of the patches.
        stride: The stride used to create the patches.
    if DEBUG:
        output_images_folder: The folder where the patches divied by rectangles will be saved.
    */

    cout << "[SR7 INFO Rebuilder] Start rebuilding image" << endl;
    cout << "[SR7 INFO Rebuilder] Found " << img_patch_vec.size() << " patches" << endl;

    int stride_pixels = patch_size * stride;
    int overlap = patch_size - stride_pixels;
    int underlap = patch_size - 2*overlap;
    cout << "[SR7 INFO Rebuilder] Patch size : " << patch_size << endl;
    cout << "[SR7 INFO Rebuilder] Stride ratio : " << stride << endl;
    cout << "[SR7 INFO Rebuilder] Stride pixels : " << stride_pixels << endl;
#if DEBUG_REBUILDER
    cout << "[SR7 INFO Rebuilder] Underlap : " << underlap << endl;
    cout << "[SR7 INFO Rebuilder] Overlap : " << overlap << endl;
    cout << "[SR7 INFO Rebuilder] Cols : " << reconstruced_image.cols << endl;
    cout << "[SR7 INFO Rebuilder] Rows : " << reconstruced_image.rows << endl;
#endif

    int n_patch_col = (reconstruced_image.cols-2*overlap)/stride_pixels;
    int n_patch_row = (reconstruced_image.rows-2*overlap)/stride_pixels;
#if DEBUG_REBUILDER
    cout << "[SR7 INFO Rebuilder] n_patch_col : " << n_patch_col << endl;
    cout << "[SR7 INFO Rebuilder] n_patch_row : " << n_patch_row << endl;
#endif

    int overlap_col = patch_size - (reconstruced_image.cols - n_patch_col*stride_pixels - overlap);
    int overlap_row = patch_size - (reconstruced_image.rows - n_patch_row*stride_pixels - overlap);
#if DEBUG_REBUILDER
    cout << "[SR7 INFO Rebuilder] overlap_col : " << overlap_col << endl;
    cout << "[SR7 INFO Rebuilder] overlap_row : " << overlap_row << endl;
#endif

    if (stride!=1.0) {
    // Regular rectangles
    Rect left(0, overlap, overlap, underlap);
    Rect top(overlap, 0, underlap, overlap);
    Rect right(underlap+overlap, overlap, overlap, underlap);
    Rect bottom(overlap, underlap+overlap, underlap, overlap);

    Rect corner_top_left(0, 0, overlap, overlap);
    Rect corner_top_right(overlap+underlap, 0, overlap, overlap);
    Rect corner_bottom_right(overlap+underlap, overlap+underlap, overlap, overlap);
    Rect corner_bottom_left(0, overlap+underlap, overlap, overlap);

    // Right penultimate edge rectangles
    Rect pnl_R_bottom(overlap, underlap+overlap, patch_size-overlap_col-overlap, overlap);
    Rect pnl_R_top(overlap, 0, patch_size-overlap_col-overlap, overlap);
    Rect pnl_R_right(patch_size-overlap_col, overlap, overlap_col, underlap);
    Rect pnl_R_corner_top_right(patch_size-overlap_col, 0, overlap_col, overlap);
    Rect pnl_R_corner_bottom_right(patch_size-overlap_col, underlap+overlap, overlap_col, overlap);

    // Right edge rectangles
    Rect R_left(0, overlap, overlap_col, underlap);
    Rect R_corner_top_left(0, 0, overlap_col, overlap);
    Rect R_corner_bottom_left(0, underlap+overlap, overlap_col, overlap);
    Rect R_corner_bottom_right(overlap_col, underlap+overlap, patch_size-overlap_col, overlap);
    Rect R_corner_top_right(overlap_col, 0, patch_size-overlap_col, overlap);

    // Bottom penultimate edge rectangles
    Rect pnl_B_bottom(overlap, patch_size-overlap_row, underlap, overlap_row);
    Rect pnl_B_corner_bottom_left(0, patch_size-overlap_row, overlap, overlap_row);
    Rect pnl_B_corner_bottom_right(underlap+overlap, patch_size-overlap_row, overlap, overlap_row);
    Rect pnl_B_right(underlap+overlap, overlap, overlap, underlap+overlap-overlap_row);
    Rect pnl_B_left(0, overlap, overlap, underlap+overlap-overlap_row);

    // Bottom edge rectangles
    Rect B_top(overlap, 0, underlap, overlap_row);
    Rect B_corner_top_left(0, 0, overlap, overlap_row);
    Rect B_corner_top_right(underlap+overlap, 0, overlap, overlap_row);
    Rect B_right(underlap+overlap, overlap_row, overlap, patch_size-overlap_row);
    Rect B_left(0, overlap_row, overlap, patch_size-overlap_row);

    // Bottom right penultimate corner
    Rect pnl_BR_right(patch_size-overlap_col, overlap, overlap_col, underlap+overlap-overlap_row);
    Rect pnl_BR_corner_bottom_right(underlap+overlap+overlap-overlap_col, patch_size-overlap_row, overlap_col, overlap_row);
    Rect pnl_BR_bottom(overlap, patch_size-overlap_row, patch_size-overlap_col-overlap, overlap_row);

    // Last penultimate corner
    Rect pnl_L_top(overlap, 0, patch_size-overlap_col-overlap, overlap_row);
    Rect pnl_L_corner_top_right(patch_size-overlap_col, 0, overlap_col, overlap_row);
    Rect pnl_L_right(patch_size-overlap_col, overlap_row, overlap_col, patch_size-overlap_row);

    // Last penultimate right corner
    Rect pnl_LR_left(0, overlap, overlap_col, underlap+overlap-overlap_row);
    Rect pnl_LR_corner_bottom_left(0, underlap+2*overlap-overlap_row, overlap_col, overlap_row);
    Rect pnl_LR_corner_bottom_right(overlap_col, underlap+2*overlap-overlap_row, patch_size-overlap_col, overlap_row);
    
    // Last corners
    Rect L_left(0, overlap_row, overlap_col, patch_size-overlap_row);
    Rect L_top(0, 0, overlap_col, overlap_row);
    Rect L_corner_top_right(overlap_col, 0, patch_size-overlap_col, overlap_row);
    }

    for (int n=0; n<name_vec.size(); n++){
        string patch_name = name_vec[n];
        Mat patch = img_patch_vec[n];

        size_t i_pos = patch_name.find("i");
        size_t j_pos = patch_name.find("j");
        size_t endi_pos = patch_name.find("endi");
        size_t endj_pos = patch_name.find("endj");

        int row = 2*stoi(patch_name.substr(i_pos + 1, endi_pos - i_pos - 1));
        int col = 2*stoi(patch_name.substr(j_pos + 1, endj_pos - j_pos - 1));

        if (stride!=1.0) {
        // First patch (top left corner)
        if ((row == 0) and (col == 0)) {
            patch(right) *= 0.5;
            patch(bottom) *= 0.5;

            patch(corner_top_right) *= 0.5;
            patch(corner_bottom_right) *= 0.25;
            patch(corner_bottom_left) *= 0.5;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Top edge patches
        else if ((row!=0) and (col==0) and (row<reconstruced_image.cols-patch_size-stride_pixels)) {
            patch(left) *= 0.5;
            patch(right) *= 0.5;
            patch(bottom) *= 0.5;
            patch(corner_top_left) *= 0.5;
            patch(corner_top_right) *= 0.5;

            patch(corner_bottom_left) *= 0.25;
            patch(corner_bottom_right) *= 0.25;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        // Left edge patches
        else if ((row==0) and (col!=0) and (col<reconstruced_image.rows-2*patch_size)){//-stride_pixels)){
            patch(top) *= 0.5;
            patch(right) *= 0.5;
            patch(bottom) *= 0.5;
            patch(corner_top_left) *= 0.5;
            patch(corner_bottom_left) *= 0.5;

            patch(corner_top_right) *= 0.25;
            patch(corner_bottom_right) *= 0.25;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // First right penultimate edge patch
        else if ((row==(n_patch_col-1)*stride_pixels) and (col==0)){
            patch(left) *= 0.5;
            patch(corner_top_left) *= 0.5;
            patch(pnl_R_bottom) *= 0.5;
            patch(pnl_R_right) *= 0.5;
            patch(pnl_R_corner_top_right) *= 0.5;

            patch(pnl_R_corner_bottom_right) *= 0.25;
            patch(corner_bottom_left) *= 0.25;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif        
        }
        // Others right penultimate edge patches
        else if ((row==(n_patch_col-1)*stride_pixels) and (col!=0) and (col<reconstruced_image.rows-2*patch_size)){
            patch(left) *= 0.5;
            patch(corner_top_left) *= 0.25;
            patch(corner_bottom_left) *= 0.25;
            patch(pnl_R_bottom) *= 0.5;
            patch(pnl_R_right) *= 0.5;
            patch(pnl_R_top) *= 0.5;
            patch(pnl_R_corner_bottom_right) *= 0.25;
            patch(pnl_R_corner_top_right) *= 0.25;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // First right edge patch
        else if ((row==reconstruced_image.cols-patch_size) and (col==0)){
            patch(R_left) *= 0.5;
            patch(R_corner_top_left) *= 0.5;
            patch(R_corner_bottom_left) *= 0.25;
            patch(R_corner_bottom_right) *= 0.5;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        // Others right edge patches
        else if ((row==reconstruced_image.cols-patch_size) and (col!=0) and (col<reconstruced_image.rows-2*patch_size)){
            patch(R_left) *= 0.5;
            patch(R_corner_top_left) *= 0.25;
            patch(R_corner_bottom_left) *= 0.25;
            patch(R_corner_bottom_right) *= 0.5;
            patch(R_corner_top_right) *= 0.5;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        // First penultimate bottom edge patch
        else if ((col==(n_patch_row-1)*stride_pixels) and (row==0)){
            patch(top) *= 0.5;
            patch(corner_top_left) *= 0.5;
            patch(corner_top_right) *= 0.25;
            
            patch(pnl_B_right) *= 0.5;
            patch(pnl_B_bottom) *= 0.5;
            patch(pnl_B_corner_bottom_left) *= 0.5;
            patch(pnl_B_corner_bottom_right) *= 0.25;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        // Others penultimate bottom edge patch
        else if ((col==(n_patch_row-1)*stride_pixels) and (row!=0) and (row<reconstruced_image.cols-2*patch_size)){
            patch(top) *= 0.5;
            patch(corner_top_left) *= 0.25;
            patch(corner_top_right) *= 0.25;
            
            patch(pnl_B_right) *= 0.5;
            patch(pnl_B_left) *= 0.5;
            patch(pnl_B_bottom) *= 0.5;
            patch(pnl_B_corner_bottom_left) *= 0.25;
            patch(pnl_B_corner_bottom_right) *= 0.25;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // First bottom edge patch
        else if ((col==reconstruced_image.rows-patch_size) and (row==0)){
            patch(B_top) *= 0.5;
            patch(B_corner_top_left) *= 0.5;
            patch(B_corner_top_right) *= 0.25;
            patch(B_right) *= 0.5;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        // Others bottom edge patch
        else if ((col==reconstruced_image.rows-patch_size) and (row!=0) and (row<reconstruced_image.cols-2*patch_size)){
            patch(B_top) *= 0.5;
            patch(B_corner_top_left) *= 0.25;
            patch(B_corner_top_right) *= 0.25;
            patch(B_left) *= 0.5;
            patch(B_right) *= 0.5;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Penultimate corner right patch
        else if ((row==(n_patch_col-1)*stride_pixels) and (col==(n_patch_row-1)*stride_pixels)){
            patch(corner_top_left) *= 0.25;
            patch(pnl_R_top) *= 0.5;
            patch(pnl_R_corner_top_right) *= 0.25;
            patch(pnl_BR_right) *= 0.5;
            patch(pnl_B_left) *= 0.5;
            patch(pnl_B_corner_bottom_left) *= 0.25;
            patch(pnl_BR_bottom) *= 0.5;
            patch(pnl_BR_corner_bottom_right) *= 0.25;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        // Last penultimate corner
        else if ((row==reconstruced_image.cols-patch_size) and (col==(n_patch_row-1)*stride_pixels)){
            patch(pnl_LR_left) *= 0.5;
            patch(R_corner_top_right) *= 0.5;
            patch(R_corner_top_left) *= 0.25;
            patch(pnl_LR_corner_bottom_left) *= 0.25;
            patch(pnl_LR_corner_bottom_right) *= 0.5;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Penultimate corner bottom patch
        else if ((row==(n_patch_col-1)*stride_pixels) and (col==reconstruced_image.rows-patch_size)){            
            patch(B_corner_top_left) *= 0.25;
            patch(B_left) *= 0.5;
            patch(pnl_L_top) *= 0.5;
            patch(pnl_L_corner_top_right) *= 0.25;
            patch(pnl_L_right) *= 0.5;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        // Last corner
        else if ((row==reconstruced_image.cols-patch_size) and (col==reconstruced_image.rows-patch_size)){
            patch(L_left) *= 0.5;
            patch(L_top) *= 0.25;
            patch(L_corner_top_right) *= 0.5;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Center patches
        else if ((row!=0) and (col!=0) and (col-stride<reconstruced_image.rows-2*patch_size) and (row<reconstruced_image.cols-2*patch_size)){
            patch(left) *= 0.5;
            patch(top) *= 0.5;
            patch(right) *= 0.5;
            patch(bottom) *= 0.5;

            patch(corner_top_left) *= 0.25;
            patch(corner_top_right) *= 0.25;
            patch(corner_bottom_right) *= 0.25;
            patch(corner_bottom_left) *= 0.25;
#if DEBUG_REBUILDER
            imwrite(output_images_folder+patch_name,patch);
#endif
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // ERROR
        else {
            cout << "[SR7 ERROR Rebuilder] Patch " << patch_name << " is not in the right position" << endl;
        }
        }
    
        // Apply patch on the reconstructed image
        Rect patch_rect(row, col, patch.rows, patch.cols);
        reconstruced_image(patch_rect) += patch;
    }
}

// int main(int argc, char const *argv[])
// {   
//     if (argc < 5) {
//         cout << "Usage: ./rebuilder <path_image> <patch_folder> <patch_size> <stride>\n";
//         return -1;
//     }
//     string path_image = argv[1];
//     string patch_folder = argv[2];
//     int patch_size = stoi(argv[3]);
//     float stride = stof(argv[4]);
    
//     // string patch_folder = "../../debug/patcher/";
//     vector<Mat> img_patch_vec;
//     vector<string> name_vec;
//     ListImages(patch_folder, name_vec, img_patch_vec);

//     Mat image = imread(path_image);
//     int IMG_WIDTH = image.cols;
//     int IMG_HEIGHT = image.rows;
//     Mat reconstructed_image(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);
    
//     cout << "[SR7 INFO Rebuilder] Found " << name_vec.size() << " patches\n";
//     rebuild_image_2(img_patch_vec, name_vec, reconstructed_image, patch_size, stride);
    
//     imwrite("../../debug/rebuilder/reconstructed_image.png", reconstructed_image);
//     return 0;
// }