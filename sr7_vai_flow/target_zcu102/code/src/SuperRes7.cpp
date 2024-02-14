#include <iostream>
#include <chrono>
#include "patcher.cpp"
#include "rebuilder.cpp"
#include "bilateral_filter.cpp"
//#include "runCNN.cpp"
#include "runCNN_test.cpp"

int main(int argc, char** argv) {

    auto start_global = std::chrono::high_resolution_clock::now();

    if (argc < 4) {
        cout << "Usage with img saving: ./SuperRes7 <image_path> <patch_size> <stride> <mask_path> <output_folder> <output_patch_folder>\n";
        cout << "Usage with w/ saving: ./SuperRes7 <image_path> <patch_size> <stride> <mask_path> <output_folder>\n";
        return -1;
    }

    cout << "\n ###################################### START #################################################\n" << endl;
    cout << "[SR7 INFO] SuperRes7 started...\n";

    /////////////////////////////////////////////////////////////////////////////////////////////
    // READING ARGUMENTS
    cout << "\n ###################################### LOADING IMAGE #########################################\n" << endl;
    string output_patch_folder;
    string image_path = argv[1];
    int patch_size = stoi(argv[2]);
    float stride = stof(argv[3]);
    string mask_path = argv[4];
    string output_folder = argv[5];
    bool save = false;
    if (argc == 7) {
         output_patch_folder = argv[6];
         save = true;
    }

    Mat image = imread(image_path);

    if (image.empty()) {
        cerr << "[SR7 ERROR] Image not found." << endl;
        return -1;
    }
    else{
        cout << "[SR7 INFO] Image successfully loaded " << image_path << " sized " << image.size() << endl;
    }

    size_t IMG_HEIGHT = image.rows;
    size_t IMG_WIDTH = image.cols;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // PATCHER
    cout << "\n ###################################### PATCHER ###############################################\n" << endl;

    auto start_patcher = std::chrono::high_resolution_clock::now();

    vector<Mat> img_patch_vec;
    vector<string> name_vec;
    if (save){
        patch_image(image, img_patch_vec, name_vec, patch_size, stride, output_patch_folder);
    }
    else{
        patch_image(image, img_patch_vec, name_vec, patch_size, stride);
    }

    auto end_patcher = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_patcher = end_patcher - start_patcher;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // SUPER RESOLUTION IA
    cout << "\n ###################################### SUPER RESOLUTION IA ###################################\n" << endl;

    auto start_IA = std::chrono::high_resolution_clock::now();

    vector<Mat> doub_img_patch_vec;
    interpolateImages(img_patch_vec, doub_img_patch_vec);
    //runCNN(img_patch_vec, doub_img_patch_vec, "/home/petalinux/target_zcu102/fsrcnn6_relu/model/fsrcnn6_relu.xmodel", 1);

    auto end_IA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_IA = end_IA - start_IA;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // REBUILDER
    cout << "\n ###################################### REBUILDER #############################################\n" << endl;

    auto start_rebuilder = std::chrono::high_resolution_clock::now();

    Mat sum_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_16UC3);
    rebuild_image(sum_image, doub_img_patch_vec, name_vec);
    
    Mat reconstructed_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8UC3);

    Mat mask(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8U);
    
    mask = imread(mask_path);
    apply_mask(sum_image, mask, reconstructed_image);

    auto end_rebuilder = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_rebuilder = end_rebuilder - start_rebuilder;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // FILTER IMAGE
    cout << "\n ###################################### FILTERING #############################################\n" << endl;

    auto start_filter = std::chrono::high_resolution_clock::now();

    Mat filtered_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8UC3);
    bilateral_filter(reconstructed_image, filtered_image);

    auto end_filter = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_filter = end_filter - start_filter;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // SAVE IMAGE
    cout << "\n ###################################### SAVING IMAGE ##########################################\n" << endl;
    sum_image.convertTo(sum_image, CV_8UC3);
    imwrite(output_folder + "sum_image.png", sum_image);
    imwrite(output_folder + "original_image.png", image);
    imwrite(output_folder + "reconstructed_image.png", reconstructed_image);
    imwrite(output_folder + "filtered_image.png", filtered_image);
    
    cout << "[SR7 INFO] Images saved in " << output_folder << endl;

    cout << "\n ###################################### END ###################################################\n" << endl;

    auto end_global = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_global = end_global - start_global;
    std::cout << "Time taken in total: " << duration_global.count() << " seconds" << std::endl;
    return 0;
}