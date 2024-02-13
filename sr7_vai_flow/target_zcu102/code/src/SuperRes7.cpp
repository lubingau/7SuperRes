#include "patcher.cpp"
#include "rebuilder.cpp"
#include "runCNN.cpp"

#define IMG_HEIGHT 1419
#define IMG_WIDTH 1672


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
    if (argc == 5) {
        outputFolder = argv[5];
        save = true;
    }

    Mat image = imread(image_path);

    if (image.empty()) {
        cerr << "Error: Image not found." << endl;
        return -1;
    }

    vector<Mat> img_patch_vec;
    vector<string> name_vec;
    if (save){
        patch_image(image, img_patch_vec, name_vec, patch_size, stride, outputFolder);
    }
    else{
        patch_image(image, img_patch_vec, name_vec, patch_size, stride);
    }

    vector<Mat> doub_img_patch_vec;
    interpolateImages(img_patch_vec, doub_img_patch_vec);

    Mat reconstructed_image(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_16UC3);
    rebuild_image(reconstructed_image, doub_img_patch_vec, name_vec);
    
    Mat mask(2 * IMG_HEIGHT, 2 * IMG_WIDTH, CV_8U);
    mask = imread(mask_path);
    mean_matrix(image, reconstructed_image, reconstructed_image);

    imwrite(outputFolder + "/original_image.png", image);
    imwrite(outputFolder + "/reconstructed_image.png", reconstructed_image);
    

    return 0;
}