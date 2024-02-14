#include "patcher.cpp"
#include "rebuilder.cpp"
#include "bilateral_filter.cpp"
//#include "runCNN.cpp"
#include "runCNN_test.cpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage with img saving: ./SuperRes7 <image_path> <patch_size> <stride> <mask_path> <output_folder>\n";
        cout << "Usage with w/ saving: ./SuperRes7 <image_path> <patch_size> <stride> <mask_path>\n";
        return -1;
    }

    string outputFolder;
    cout << "\n ###################################### START #################################################\n" << endl;
    cout << "[SR7 INFO] SuperRes7 started...\n";
    /////////////////////////////////////////////////////////////////////////////////////////////
    // READING ARGUMENTS
    cout << "\n ###################################### LOADING IMAGE #########################################\n" << endl;
    string image_path = argv[1];
    int patch_size = stoi(argv[2]);
    float stride = stof(argv[3]);
    string mask_path = argv[4];
    bool save = false;
    if (argc == 6) {
         outputFolder = argv[5];
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
    vector<Mat> img_patch_vec;
    vector<string> name_vec;

    if (save){
        patch_image(image, img_patch_vec, name_vec, patch_size, stride, outputFolder);
    }
    else{
        patch_image(image, img_patch_vec, name_vec, patch_size, stride);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // SUPER RESOLUTION IA
    cout << "\n ###################################### SUPER RESOLUTION IA ###################################\n" << endl;
    vector<Mat> doub_img_patch_vec;
    interpolateImages(img_patch_vec, doub_img_patch_vec);
    //runCNN(img_patch_vec, doub_img_patch_vec, "/home/petalinux/target_zcu102/fsrcnn6_relu/model/fsrcnn6_relu.xmodel", 1);

    /////////////////////////////////////////////////////////////////////////////////////////////
    // REBUILDER
    cout << "\n ###################################### REBUILDER #############################################\n" << endl;
    Mat sum_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_16UC3);
    rebuild_image(sum_image, doub_img_patch_vec, name_vec);
    
    Mat reconstructed_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8UC3);

    Mat mask(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8U);
    
    mask = imread(mask_path);
    apply_mask(sum_image, mask, reconstructed_image);


    /////////////////////////////////////////////////////////////////////////////////////////////
    // FILTER IMAGE
    cout << "\n ###################################### FILTERING #############################################\n" << endl;
    Mat filtered_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8UC3);
    bilateral_filter(reconstructed_image, filtered_image);

    /////////////////////////////////////////////////////////////////////////////////////////////
    // SAVE IMAGE
    cout << "\n ###################################### SAVING IMAGE ##########################################\n" << endl;
    sum_image.convertTo(sum_image, CV_8UC3);
    imwrite(outputFolder + "sum_image.png", sum_image);
    imwrite(outputFolder + "original_image.png", image);
    imwrite(outputFolder + "reconstructed_image.png", reconstructed_image);
    imwrite(outputFolder + "filtered_image.png", filtered_image);
    cout << "[SR7 INFO] Images saved in " << outputFolder << endl;

    cout << "\n ###################################### END ###################################################\n" << endl;
    return 0;
}