#include <iostream>
#include <chrono>
#include "patcher.cpp"
#include "rebuilder.cpp"
#include "bilateral_filter.cpp"
#include "runCNN.cpp"
//#include "runCNN_test.cpp"

#include "debug.h"


int main(int argc, char** argv) {

    auto start_global = std::chrono::high_resolution_clock::now();

    if (argc < 5) {
        cout << "Usage: ./SuperRes7 <image_path> <patch_size> <stride> <path_xmodel> <output_folder>\n";
        cout << "Usage with debug: ./SuperRes7 <image_path> <patch_size> <stride> <path_xmodel> <output_folder> <debug_folder>\n";
        return -1;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////
    // READING ARGUMENTS
    string image_path = argv[1];
    int patch_size = stoi(argv[2]);
    float stride = stof(argv[3]);
    int stride_pixels = patch_size * stride;
    string path_xmodel = argv[4];
    string output_folder = argv[5];
    string debug_folder = argv[6];

    cout << "\n###################################### START #################################################\n" << endl;
    cout << "[SR7 INFO] SuperRes7 started\n";

    cout << "\n###################################### LOADING IMAGE #########################################\n" << endl;

    Mat image = imread(image_path);

    if (image.empty()) {
        cerr << "[SR7 ERROR] Image not found." << endl;
        return -1;
    }
    else{
        cout << "[SR7 INFO] Image successfully loaded from " << image_path << endl;
        cout << "[SR7 INFO] Image size: " << image.rows << "x" << image.cols << endl;
    }

    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_load = end_load - start_global;

    size_t IMG_HEIGHT = image.rows;
    size_t IMG_WIDTH = image.cols;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // PATCHER
    cout << "\n###################################### PATCHER ###############################################\n" << endl;

    vector<Mat> img_patch_vec;
    vector<string> name_vec;
#if DEBUG_PATCHER
    string debug_patcher_folder = debug_folder + "patcher/";
    patch_image(image, img_patch_vec, name_vec, patch_size, stride, debug_patcher_folder);
#else
    patch_image(image, img_patch_vec, name_vec, patch_size, stride);
#endif
    image.release();

    auto end_patcher = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_patcher = end_patcher - end_load;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // SUPER RESOLUTION AI
    cout << "\n###################################### SUPER RESOLUTION AI ###################################\n" << endl;
    cout << "[SR7 INFO AI] Starting super-resolution AI" << endl;

    vector<Mat> doub_img_patch_vec;
    vector<Mat> img_patch_vec_temp;
    vector<Mat> doub_img_patch_vec_temp;
    int threads = 6;

    for (int n = 0; n < img_patch_vec.size(); n++){
        Mat temp = img_patch_vec[n];
        img_patch_vec_temp.push_back(temp);
        if (((n==img_patch_vec.size()-1)or(n%2000-1==0)) and (n>1)){
            cout << "\n[SR7 INFO AI] " << img_patch_vec_temp.size() << " patches into DPU\n" << endl;
#if DEBUG_RUNCNN
            string debug_runCNN_input_folder = debug_folder + "runCNN/input/";
            string debug_runCNN_output_folder = debug_folder + "runCNN/output/";
            runCNN(img_patch_vec_temp, doub_img_patch_vec_temp, path_xmodel, threads, debug_runCNN_input_folder, debug_runCNN_output_folder);
#else
            runCNN(img_patch_vec_temp, doub_img_patch_vec_temp, path_xmodel, threads);
            //extrapolateImages(img_patch_vec_temp, doub_img_patch_vec_temp);
#endif
            for (int i = 0; i < doub_img_patch_vec_temp.size(); i++){
                doub_img_patch_vec.push_back(doub_img_patch_vec_temp[i]);
            }
            img_patch_vec_temp.clear();
            img_patch_vec_temp.shrink_to_fit();
            doub_img_patch_vec_temp.clear();
            doub_img_patch_vec_temp.shrink_to_fit();
            cout << "DEBUG: doub_img_patch_vec " << doub_img_patch_vec.size() << " size after clear()" << endl;
            cout << "DEBUG: img_patch_vec_temp " << img_patch_vec_temp.size() << " size after clear()" << endl;
        }
    }
    img_patch_vec.clear();

#if DEBUG_RUNCNN
    // Only for debug
    for (int n=0; n<doub_img_patch_vec.size(); n++){
        Mat debug = doub_img_patch_vec[n];
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                int B_pix = debug.at<Vec3b>(i, j)[0];
                int G_pix = debug.at<Vec3b>(i, j)[1];
                int R_pix = debug.at<Vec3b>(i, j)[2];
                //cout << "B: " << B_pix << " G: " << G_pix << " R: " << R_pix << endl;
            }
        }
        string filename = debug_runCNN_output_folder + "debug_out_" + to_string(n) + ".png";
        imwrite(filename, debug);
        cout << "[SR7 INFO] Image " << n << " super-resolved" << endl;
    }
#endif
    cout << "[SR7 INFO] All patches super-resolved" << endl;

    auto end_IA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_IA = end_IA - end_patcher;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // REBUILDER
    cout << "\n###################################### REBUILDER #############################################\n" << endl;

    Mat reconstructed_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_16UC3);
#if DEBUG_REBUILDER
    string debug_rebuilder_folder = debug_folder + "rebuilder/";
    rebuild_image_and_mask(doub_img_patch_vec, name_vec, reconstructed_image, debug_rebuilder_folder);
#else
    rebuild_image_and_mask(doub_img_patch_vec, name_vec, reconstructed_image);
#endif
    reconstructed_image.convertTo(reconstructed_image, CV_8UC3);
    doub_img_patch_vec.clear();
    name_vec.clear();

    auto end_rebuilder = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_rebuilder = end_rebuilder - end_IA;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // FILTER IMAGE
    // cout << "\n###################################### FILTERING #############################################\n" << endl;

    // Mat filtered_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8UC3);
    // bilateral_filter(reconstructed_image, filtered_image);
    // reconstructed_image.release();

    auto end_filter = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_filter = end_filter - end_rebuilder;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // SAVE IMAGE
    cout << "\n###################################### SAVING IMAGE ##########################################\n" << endl;

    imwrite(output_folder + "reconstructed_image.png", reconstructed_image);
    reconstructed_image.release();
    
    cout << "[SR7 INFO] Images saved in " << output_folder << endl;

    auto end_save = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_save = end_save - end_filter;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // EXECUTION TIMES
    cout << "\n###################################### EXECUTION TIMES #######################################\n" << endl;

    auto end_global = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_global = end_global - start_global;

    //Execution time
    std::cout << "[SR7 INFO] EXECUTION TIMES:" << std::endl;
    std::cout << " __________________________________________________________________________________________________" << std::endl;
    std::cout << "|    Load      |    Patcher   |     AI    |   Rebuilder  |     Filter    |     Save    |   Total   |" << std::endl;
    std::cout << "|     " << std::fixed << std::setprecision(3) <<  duration_load.count() << "s   |    "<<  duration_patcher.count() << "s    |   " << duration_IA.count() << "s  |     " << duration_rebuilder.count() << "s   |     " << duration_filter.count() << "s    |    " <<  duration_save.count() << "s   |   " << duration_global.count() << "s  |" << std::endl;
    std::cout << " ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾" << std::endl;

    cout << "\n###################################### END ###################################################\n" << endl;

    return 0;
}