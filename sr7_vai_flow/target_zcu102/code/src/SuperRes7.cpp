#include "patcher.cpp"
#include "rebuilder.cpp"
#include "runCNN.cpp"
//#include "runCNN_test.cpp"


int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage with img saving: ./SuperRes7 <image_path> <patch_size> <stride> <mask_path> <output_folder>\n";
        cout << "Usage with w/ saving: ./SuperRes7 <image_path> <patch_size> <stride> <mask_path>\n";
        return -1;
    }

    string outputFolder;

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
        cerr << "Error: Image not found." << endl;
        return -1;
    }
    else{
        cout << "[SR7 INFO] Sucessfully loaded " << image_path << " sized " << image.size() << endl;
    }

    // if (image.rows % 2 != 0) {
    //     cout << "[SR7 WARNING] The image height is not even. Last pixels deleted" << endl;
    //     image = image(Rect(0, 0, image.cols, image.rows - 1));
    // }
    // if (image.cols % 2 != 0) {
    //     cout << "[SR7 WARNING] The image width is not even. Last pixels deleted" << endl;
    //     image = image(Rect(0, 0, image.cols - 1, image.rows));
    // }

    size_t IMG_HEIGHT = image.rows;
    size_t IMG_WIDTH = image.cols;

    vector<Mat> img_patch_vec;
    vector<string> name_vec;
    /////////////////////////////////////////////////////////////////////////////////////////////
    // PATCHER
    cout << "######### PATCHER #########\n"
    if (save){
        patch_image(image, img_patch_vec, name_vec, patch_size, stride, outputFolder);
    }
    else{
        patch_image(image, img_patch_vec, name_vec, patch_size, stride);
    }

    vector<Mat> doub_img_patch_vec;
    interpolateImages(img_patch_vec, doub_img_patch_vec);
    //runCNN(img_patch_vec, doub_img_patch_vec, "/home/petalinux/target_zcu102/fsrcnn6_relu/model/fsrcnn6_relu.xmodel", 1);

    Mat sum_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_16UC3);
    rebuild_image(sum_image, doub_img_patch_vec, name_vec);


    
    Mat reconstructed_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8UC3);

    Mat mask(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8U);
    
    mask = imread(mask_path);
    apply_mask(sum_image, mask, reconstructed_image);

    sum_image.convertTo(sum_image, CV_8UC3);
    imwrite(outputFolder + "sum_image.png", sum_image);

    imwrite(outputFolder + "original_image.png", image);
    imwrite(outputFolder + "reconstructed_image.png", reconstructed_image);
    

    return 0;
}