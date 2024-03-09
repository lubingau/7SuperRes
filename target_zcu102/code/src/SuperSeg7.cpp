#include <iostream>
#include <fstream>
#include <chrono>
#include "patcher.cpp"
#include "rebuilder.cpp"
#include "runCNN.cpp"
#include "mask2mat.cpp"

#include "debug.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

int main(int argc, char** argv) {

    /*
    Main function of the SuperRes7 application. This function reads the input arguments, loads the input image, 
    patches it, runs the super-resolution AI, rebuilds the image and saves it.

    Args:
        argc: Number of input arguments
        argv: Input arguments
    */

    auto start_global = high_resolution_clock::now();

    if (argc < 8) {
        cout << "Usage: ./SuperSeg7 <image_path> <patch_size> <stride> <path_xmodel_segment> <output_folder> <threads> <processes> <debug_folder>" << endl;
        cout << "WARNING: <debug_folder> is optional" << endl;
        return -1;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////
    // READING ARGUMENTS
    string image_path = argv[1];
    int patch_size = stoi(argv[2]);
    float stride = stof(argv[3]);
    int stride_pixels = patch_size * stride;
    int overlap = patch_size - stride_pixels;
    string path_xmodel_segment = argv[4];
    string output_folder = argv[5];
    int num_threads = stoi(argv[6]);
    int num_processes = stoi(argv[7]);
    string debug_folder = argv[8];

    /////////////////////////////////////////////////////////////////////////////////////////////
    // LOADING IMAGE

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

    auto end_load = high_resolution_clock::now();
    duration<double> duration_load = end_load - start_global;

    //////////////////////////////////////////////////////////////////////////
    // BUFFERS INITIALIZATION
    //////////////////////////////////////////////////////////////////////////

    size_t IMG_WIDTH = image.cols;
    size_t IMG_HEIGHT = image.rows;

    int n_patches_i = (IMG_WIDTH - 2*overlap)/(stride_pixels);
    int n_patches_j = (IMG_HEIGHT - 2*overlap)/(stride_pixels);

    int n_patches = n_patches_i * n_patches_j + n_patches_i + n_patches_j + 1;
    int inSize = patch_size * patch_size * image.channels();
    int num_of_classes = 7;
    int outSize = patch_size * patch_size * num_of_classes;

    cout << "[SR7 INFO] width: " << IMG_WIDTH << endl;
    cout << "[SR7 INFO] height: " << IMG_HEIGHT << endl;
    cout << "[SR7 INFO] n_patches_i (cols): " << n_patches_i << endl;
    cout << "[SR7 INFO] n_patches_j (rows): " << n_patches_j << endl;
    cout << "[SR7 INFO] n_patches: " << n_patches << endl;

    // Colors for the classes (BGR format)
    vector<Vec3b> colors;
    colors.push_back(Vec3b(255, 255, 0));    // urban_land
    colors.push_back(Vec3b(0, 255, 255));    // agriculture
    colors.push_back(Vec3b(255, 0, 255));    // rangeland
    colors.push_back(Vec3b(0, 255, 0));      // forest_land
    colors.push_back(Vec3b(255, 0, 0));      // water
    colors.push_back(Vec3b(255, 255, 255));  // barren_land
    colors.push_back(Vec3b(0, 0, 0));        // unknown
    for (int i=0; i<colors.size(); i++){
        colors[i] = Vec3b(colors[i][0]/2, colors[i][1]/2, colors[i][2]/2);
    }

    int8_t *inputBuffer = new int8_t[n_patches*inSize];
    int16_t *posBuffer = new int16_t[n_patches*2];


    /////////////////////////////////////////////////////////////////////////////////////////////
    // PATCHER
    /////////////////////////////////////////////////////////////////////////////////////////////
    cout << "\n-------- PATCHER --------\n" << endl;

    cout << "[SR7 INFO] Total number of patches: " << n_patches << endl;

#if DEBUG_PATCHER
    string debug_patcher_folder = debug_folder + "patcher/";
    patch_image(image, inputBuffer, posBuffer, patch_size, stride, n_patches, debug_patcher_folder);
#else
    patch_image(image, inputBuffer, posBuffer, patch_size, stride, n_patches);
#endif
    image.release();

    auto end_patcher = high_resolution_clock::now();
    duration<double> duration_patcher = end_patcher - end_load;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // MULTI-PROCESSING
    /////////////////////////////////////////////////////////////////////////////////////////////
    cout << "\n-------- MULTI-PROCESSING --------\n" << endl;
    
    int n_patches_x_process = n_patches / num_processes;
    int n_patches_first_process = n_patches_x_process + n_patches % num_processes;
    cout << "[SR7 INFO] n_patches: " << n_patches << endl;
    cout << "[SR7 INFO] n_patches_first_process: " << n_patches_first_process << endl;
    cout << "[SR7 INFO] n_patches_x_process: " << n_patches_x_process << endl;

    int8_t *inputBuffer0, *inputBuffer1, *inputBuffer2, *inputBuffer3, *inputBuffer4, *inputBuffer5, *inputBuffer6, *inputBuffer7;
    int16_t *posBuffer0, *posBuffer1, *posBuffer2, *posBuffer3, *posBuffer4, *posBuffer5, *posBuffer6, *posBuffer7;
    int8_t *outputBuffer0 = new int8_t[outSize * n_patches_first_process];
    int8_t *outputBuffer1 = new int8_t[outSize * n_patches_x_process];
    int8_t *outputBuffer2 = new int8_t[outSize * n_patches_x_process];
    int8_t *outputBuffer3 = new int8_t[outSize * n_patches_x_process];
    int8_t *outputBuffer4 = new int8_t[outSize * n_patches_x_process];
    int8_t *outputBuffer5 = new int8_t[outSize * n_patches_x_process];
    int8_t *outputBuffer6 = new int8_t[outSize * n_patches_x_process];
    int8_t *outputBuffer7 = new int8_t[outSize * n_patches_x_process];


    if (num_processes >= 1){
        inputBuffer0 = inputBuffer;
        posBuffer0 = posBuffer;
    }
    if (num_processes >= 2){
        inputBuffer1 = inputBuffer + n_patches_first_process*inSize;
        posBuffer1 = posBuffer + n_patches_first_process*2;
    }
    if (num_processes >= 3){
        inputBuffer2 = inputBuffer + (n_patches_first_process + n_patches_x_process)*inSize;
        posBuffer2 = posBuffer + (n_patches_first_process + n_patches_x_process)*2;
    }
    if (num_processes >= 4){
        inputBuffer3 = inputBuffer + (n_patches_first_process + 2*n_patches_x_process)*inSize;
        posBuffer3 = posBuffer + (n_patches_first_process + 2*n_patches_x_process)*2;
    }
    if (num_processes >= 5){
        inputBuffer4 = inputBuffer + (n_patches_first_process + 3*n_patches_x_process)*inSize;
        posBuffer4 = posBuffer + (n_patches_first_process + 3*n_patches_x_process)*2;
    }
    if (num_processes >= 6){
        inputBuffer5 = inputBuffer + (n_patches_first_process + 4*n_patches_x_process)*inSize;
        posBuffer5 = posBuffer + (n_patches_first_process + 4*n_patches_x_process)*2;
    }
    if (num_processes >= 7){
        inputBuffer6 = inputBuffer + (n_patches_first_process + 5*n_patches_x_process)*inSize;
        posBuffer6 = posBuffer + (n_patches_first_process + 5*n_patches_x_process)*2;
    }
    if (num_processes >= 8){
        inputBuffer7 = inputBuffer + (n_patches_first_process + 6*n_patches_x_process)*inSize;
        posBuffer7 = posBuffer + (n_patches_first_process + 6*n_patches_x_process)*2;
    }

    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // CNN + REBUILDER
    /////////////////////////////////////////////////////////////////////////////////////////////

    Mat reconstructed_image = Mat::zeros(2*IMG_HEIGHT, 2*IMG_WIDTH, CV_8UC3);
#if DEBUG_REBUILDER  
    string debug_rebuilder_folder = debug_folder + "rebuilder/";
    rebuild_image(outputBuffer, posBuffer, reconstructed_image, patch_size, stride, n_patches, debug_rebuilder_folder);
#else
    auto duration_runCNN = duration<double>::zero();
    auto duration_maskToBGR = duration<double>::zero();
    auto duration_rebuilder = duration<double>::zero();

    if (num_processes >= 1){
        auto start_runCNN = high_resolution_clock::now();
        runCNN(inputBuffer0, outputBuffer0, path_xmodel_segment, num_threads, n_patches_first_process);
        duration_runCNN += high_resolution_clock::now() - start_runCNN;

        auto start_maskToBGR = high_resolution_clock::now();
        maskToBGR(outputBuffer0, inputBuffer0, n_patches_first_process, patch_size, num_of_classes, colors);
        duration_maskToBGR += high_resolution_clock::now() - start_maskToBGR;

        auto start_rebuild = high_resolution_clock::now();
        rebuild_image(inputBuffer0, posBuffer0, reconstructed_image, patch_size, stride, n_patches_first_process);
        duration_rebuilder += high_resolution_clock::now() - start_rebuild;

        delete[] outputBuffer0;
    }
    if (num_processes >= 2){
        auto start_runCNN = high_resolution_clock::now();
        runCNN(inputBuffer1, outputBuffer1, path_xmodel_segment, num_threads, n_patches_x_process);
        duration_runCNN += high_resolution_clock::now() - start_runCNN;

        auto start_rebuild = high_resolution_clock::now();
        rebuild_image(outputBuffer1, posBuffer1, reconstructed_image, patch_size, stride, n_patches_x_process);
        duration_rebuilder += high_resolution_clock::now() - start_rebuild;

        delete[] outputBuffer1;
    }
    if (num_processes >= 3){
        auto start_runCNN = high_resolution_clock::now();
        runCNN(inputBuffer2, outputBuffer2, path_xmodel_segment, num_threads, n_patches_x_process);
        duration_runCNN += high_resolution_clock::now() - start_runCNN;

        auto start_rebuild = high_resolution_clock::now();
        rebuild_image(outputBuffer2, posBuffer2, reconstructed_image, patch_size, stride, n_patches_x_process);
        duration_rebuilder += high_resolution_clock::now() - start_rebuild;

        delete[] outputBuffer2;
    }
    if (num_processes >= 4){
        auto start_runCNN = high_resolution_clock::now();
        runCNN(inputBuffer3, outputBuffer3, path_xmodel_segment, num_threads, n_patches_x_process);
        duration_runCNN += high_resolution_clock::now() - start_runCNN;

        auto start_rebuild = high_resolution_clock::now();
        rebuild_image(outputBuffer3, posBuffer3, reconstructed_image, patch_size, stride, n_patches_x_process);
        duration_rebuilder += high_resolution_clock::now() - start_rebuild;

        delete[] outputBuffer3;
    }
    if (num_processes >= 5){
        auto start_runCNN = high_resolution_clock::now();
        runCNN(inputBuffer4, outputBuffer4, path_xmodel_segment, num_threads, n_patches_x_process);
        duration_runCNN += high_resolution_clock::now() - start_runCNN;

        auto start_rebuild = high_resolution_clock::now();
        rebuild_image(outputBuffer4, posBuffer4, reconstructed_image, patch_size, stride, n_patches_x_process);
        duration_rebuilder += high_resolution_clock::now() - start_rebuild;

        delete[] outputBuffer4;
    }
    if (num_processes >= 6){
        auto start_runCNN = high_resolution_clock::now();
        runCNN(inputBuffer5, outputBuffer5, path_xmodel_segment, num_threads, n_patches_x_process);
        duration_runCNN += high_resolution_clock::now() - start_runCNN;

        auto start_rebuild = high_resolution_clock::now();
        rebuild_image(outputBuffer5, posBuffer5, reconstructed_image, patch_size, stride, n_patches_x_process);
        duration_rebuilder += high_resolution_clock::now() - start_rebuild;

        delete[] outputBuffer5;
    }
    if (num_processes >= 7){
        auto start_runCNN = high_resolution_clock::now();
        runCNN(inputBuffer6, outputBuffer6, path_xmodel_segment, num_threads, n_patches_x_process);
        duration_runCNN += high_resolution_clock::now() - start_runCNN;

        auto start_rebuild = high_resolution_clock::now();
        rebuild_image(outputBuffer6, posBuffer6, reconstructed_image, patch_size, stride, n_patches_x_process);
        duration_rebuilder += high_resolution_clock::now() - start_rebuild;

        delete[] outputBuffer6;
    }
    if (num_processes >= 8){
        auto start_runCNN = high_resolution_clock::now();
        runCNN(inputBuffer7, outputBuffer7, path_xmodel_segment, num_threads, n_patches_x_process);
        duration_runCNN += high_resolution_clock::now() - start_runCNN;

        auto start_rebuild = high_resolution_clock::now();
        rebuild_image(outputBuffer7, posBuffer7, reconstructed_image, patch_size, stride, n_patches_x_process);
        duration_rebuilder += high_resolution_clock::now() - start_rebuild;

        delete[] outputBuffer7;
    }
#endif

    delete[] inputBuffer;
    delete[] posBuffer;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // EXECUTION TIMES
    cout << "\n###################################### EXECUTION TIMES #######################################\n" << endl;


    //Execution time
    cout << "[SR7 INFO] EXECUTION TIMES:" << endl;
    cout << " ____________________________________________________________________________" << endl;
    cout << "|    Load      |    Patcher   |     CNN     |    Mask2BGR   |    Rebuilder   |" << endl;
    cout << "|     " << fixed << setprecision(3) <<  duration_load.count() << "s   |    "<<  duration_patcher.count() << "s    |    " << duration_runCNN.count() << "s   |   " << duration_maskToBGR.count() << "s   |    " << duration_rebuilder.count() << "s    |" << endl;
    cout << " ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾" << endl;
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // SAVE IMAGE
    cout << "\n###################################### SAVING IMAGE ##########################################\n" << endl;

    auto start_save = high_resolution_clock::now();
    imwrite(output_folder + "/reconstructed_image.png", reconstructed_image);
    reconstructed_image.release();
    
    cout << "[SR7 INFO] Images saved in " << output_folder << endl;

    duration<double> duration_save = high_resolution_clock::now() - start_save;
    duration<double> duration_global = high_resolution_clock::now() - start_global;

    cout << "\n[SR7 INFO] Writing time: " << duration_save.count() << "s" << endl;
    cout << "[SR7 INFO] Total execution time: " << duration_global.count() << "s" << endl;

    ofstream time_file;
    time_file.open("execution_times.txt");
    time_file << "Load Patcher CNN Mask2BGR Rebuilder Save Total" << endl;
    time_file << duration_load.count() << " " << duration_patcher.count() << " " << duration_runCNN.count() << " " << duration_maskToBGR.count() << " " << duration_rebuilder.count() << " " << duration_save.count() << " " << duration_global.count() << endl;
    time_file.close();

    cout << "[SR7 INFO] Execution times saved in execution_times.txt" << endl;

    cout << "\n###################################### END ###################################################\n" << endl;

    return 0;
}