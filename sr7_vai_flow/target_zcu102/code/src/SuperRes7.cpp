#include <iostream>
#include <chrono>
#include "patcher.cpp"
#include "rebuilder.cpp"
#include "bilateral_filter.cpp"
#include "runCNN.cpp"
//#include "runCNN_test.cpp"

int main(int argc, char** argv) {

    auto start_global = std::chrono::high_resolution_clock::now();

    if (argc < 5) {
        cout << "Usage with img saving: ./SuperRes7 <image_path> <patch_size> <stride> <output_folder> <output_patch_folder>\n";
        cout << "Usage with w/ saving: ./SuperRes7 <image_path> <patch_size> <stride> <output_folder>\n";
        return -1;
    }

    cout << "\n###################################### START #################################################\n" << endl;
    cout << "[SR7 INFO] SuperRes7 started...\n";

    /////////////////////////////////////////////////////////////////////////////////////////////
    // READING ARGUMENTS
    cout << "\n###################################### LOADING IMAGE #########################################\n" << endl;
    string output_patch_folder;
    string image_path = argv[1];
    int patch_size = stoi(argv[2]);
    float stride = stof(argv[3]);
    string output_folder = argv[4];
    bool save = false;
    if (argc == 7) {
         output_patch_folder = argv[5];
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
    cout << "\n###################################### PATCHER ###############################################\n" << endl;

    auto start_patcher = std::chrono::high_resolution_clock::now();

    vector<Mat> img_patch_vec;
    vector<string> name_vec;
    if (save){
        patch_image(image, img_patch_vec, name_vec, patch_size, stride, output_patch_folder);
    }
    else{
        patch_image(image, img_patch_vec, name_vec, patch_size, stride);
    }
    //image.release();

    auto end_patcher = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_patcher = end_patcher - start_patcher;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // SUPER RESOLUTION IA
    cout << "\n###################################### SUPER RESOLUTION IA ###################################\n" << endl;

    auto start_IA = std::chrono::high_resolution_clock::now();

    vector<Mat> doub_img_patch_vec;
    
    vector<Mat> img_patch_vec_temp;
    vector<Mat> doub_img_patch_vec_temp;
    for (int n = 0; n < img_patch_vec.size(); n++){
        Mat temp;
        img_patch_vec[n].convertTo(temp, CV_16UC3);
        img_patch_vec_temp.push_back(temp);
        if ((n%1000==0)or(n==img_patch_vec.size()-1)){
            cout << "\x1b[A";
            cout << "[SR7 INFO IA] " << n+1 << " patches into DPU" << endl;
            //runCNN(img_patch_vec_temp, doub_img_patch_vec_temp, "/home/petalinux/target_zcu102/fsrcnn6_relu/model/fsrcnn6_relu.xmodel", 2);
            interpolateImages(img_patch_vec_temp, doub_img_patch_vec_temp);
            for (int i = 0; i < doub_img_patch_vec_temp.size(); i++){
                doub_img_patch_vec.push_back(doub_img_patch_vec_temp[i]);
            }
            img_patch_vec_temp.clear();
            doub_img_patch_vec_temp.clear();
        }
    }
    //runCNN(img_patch_vec, doub_img_patch_vec, "/home/petalinux/target_zcu102/fsrcnn6_relu/model/fsrcnn6_relu.xmodel", 1);

    img_patch_vec.clear();

    Mat debug = doub_img_patch_vec[0];
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            int B_pix = debug.at<Vec3b>(i, j)[0];
            int G_pix = debug.at<Vec3b>(i, j)[1];
            int R_pix = debug.at<Vec3b>(i, j)[2];
            cout << "B: " << B_pix << " G: " << G_pix << " R: " << R_pix << endl;
        }
    }

    auto end_IA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_IA = end_IA - start_IA;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // REBUILDER
    cout << "\n###################################### REBUILDER #############################################\n" << endl;

    auto start_rebuilder = std::chrono::high_resolution_clock::now();

    Mat reconstructed_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_16UC3);
    rebuild_image_and_mask(reconstructed_image, doub_img_patch_vec, name_vec);
    doub_img_patch_vec.clear();
    name_vec.clear();

    auto end_rebuilder = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_rebuilder = end_rebuilder - start_rebuilder;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // FILTER IMAGE
    cout << "\n###################################### FILTERING #############################################\n" << endl;

    auto start_filter = std::chrono::high_resolution_clock::now();

    // Mat filtered_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8UC3);
    // //bilateral_filter(reconstructed_image, filtered_image);
    // reconstructed_image.release();

    auto end_filter = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_filter = end_filter - start_filter;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // SAVE IMAGE
    cout << "\n###################################### SAVING IMAGE ##########################################\n" << endl;

    imwrite(output_folder + "original_image.png", image);
    imwrite(output_folder + "reconstructed_image.png", reconstructed_image);
    
    cout << "[SR7 INFO] Images saved in " << output_folder << endl;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // EXECUTION TIMES
    cout << "\n###################################### EXECUTION TIMES #######################################\n" << endl;

    auto end_global = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_global = end_global - start_global;

    //Execution time
    std::cout << "[SR7 INFO] EXECUTION TIMES:" << std::endl;
    std::cout << " _____________________________________________________________________" << std::endl;
    std::cout << "|    Patcher   |     IA    |   Rebuilder  |     Filter    |   Total   |" << std::endl;
    std::cout << "|     " << std::fixed << std::setprecision(3) << duration_patcher.count() << "s   |   " << duration_IA.count() << "s  |     " << duration_rebuilder.count() << "s   |     " << duration_filter.count() << "s    |   " << duration_global.count() << "s  |" << std::endl;
    std::cout << " ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾" << std::endl;
    
    cout << "\n###################################### END ###################################################\n" << endl;

    return 0;
}