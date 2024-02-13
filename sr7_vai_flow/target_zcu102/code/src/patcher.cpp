#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

void patch_image(const Mat& image, vector<Mat>& image_arr, vector<string>& name_arr,int patch_size, float stride, const string& folder) {
    int stride_pixels = static_cast<int>(stride * patch_size);
    int overlap = patch_size - stride_pixels;
    int count = 0;

    cout << endl;

    for (int i = 0; i < image.rows - stride_pixels; i += stride_pixels) {
        for (int j = 0; j < image.cols - stride_pixels; j += stride_pixels) {
            Rect patch_rect(j, i, patch_size, patch_size);
            Mat patch = image(patch_rect).clone();

            string name = "i" + to_string(j) + "endi_j" + to_string(i) + "endj";

            image_arr.push_back(patch);
            name_arr.push_back(name);
            
            string filename = folder + name + ".png";
            imwrite(filename, patch);

            count++;
            if (count % 100 == 0){
            cout << "\x1b[A";
            cout << "[SR7 INFO] Patches created: " << count << endl;
            }     
        }
    }

    cout << "[SR7 INFO] Regular patches created: " << count << endl;

    int last_patch_j = image.cols - patch_size;
    int last_patch_i = image.rows - patch_size;

    for (int i = 0; i < image.rows - patch_size; i += stride_pixels) {
        Rect patch_rect(last_patch_j, i, patch_size, patch_size);
        Mat patch = image(patch_rect).clone();

        string name = "i" + to_string(last_patch_j) + "endi_j" + to_string(i) + "endj";
            
        image_arr.push_back(patch);
        name_arr.push_back(name);
        
        string filename = folder + name + ".png";
        imwrite(filename, patch);

        count++; 
    }

    for (int j = 0; j < image.cols - patch_size; j += stride_pixels) {
        Rect patch_rect(j, last_patch_i, patch_size, patch_size);
        Mat patch = image(patch_rect).clone();
        
        string name = "i" + to_string(j) + "endi_j" + to_string(last_patch_i) + "endj";
            
        image_arr.push_back(patch);
        name_arr.push_back(name);
        
        string filename = folder + name + ".png";
        imwrite(filename, patch);
    }

    Rect patch_rect(last_patch_j, last_patch_i, patch_size, patch_size);
    Mat patch = image(patch_rect).clone();
    
    string name = "i" + to_string(last_patch_j) + "endi_j" + to_string(last_patch_i) + "endj";
            
    image_arr.push_back(patch);
    name_arr.push_back(name);
    
    string filename = folder + name + ".png";
    imwrite(filename, patch);

    cout << "[SR7 INFO] Edge patches created";
    cout << "[SR7 INFO] Total patches created: " << count << endl;
}

void patch_image(const Mat& image, vector<Mat>& image_arr, vector<string>& name_arr,int patch_size, float stride) {
    int stride_pixels = static_cast<int>(stride * patch_size);
    int overlap = patch_size - stride_pixels;
    
    int count = 0;
    cout << endl;

    for (int i = 0; i < image.rows - stride_pixels; i += stride_pixels) {
        for (int j = 0; j < image.cols - stride_pixels; j += stride_pixels) {
            Rect patch_rect(j, i, patch_size, patch_size);
            Mat patch = image(patch_rect).clone();
            string name = "i" + to_string(j) + "endi_j" + to_string(i) + "endj";
            image_arr.push_back(patch);
            name_arr.push_back(name);
            count++;
            if (count % 100 == 0){
                cout << "\x1b[A";
                cout << "[SR7 INFO] Patches created: " << count << endl;
            }
        }
    }
    cout << "[SR7 INFO] Regular patches created: " << count << endl;

    int last_patch_j = image.cols - patch_size;
    int last_patch_i = image.rows - patch_size;

    for (int i = 0; i < image.rows - patch_size; i += stride_pixels) {
        Rect patch_rect(last_patch_j, i, patch_size, patch_size);
        Mat patch = image(patch_rect).clone();
        string name = "i" + to_string(last_patch_j) + "endi_j" + to_string(i) + "endj";
        image_arr.push_back(patch);
        name_arr.push_back(name);
        count++;
    }

    for (int j = 0; j < image.cols - patch_size; j += stride_pixels) {
        Rect patch_rect(j, last_patch_i, patch_size, patch_size);
        Mat patch = image(patch_rect).clone();
        string name = "i" + to_string(j) + "endi_j" + to_string(last_patch_i) + "endj";
        image_arr.push_back(patch);
        name_arr.push_back(name);
        count++;
    }

    Rect patch_rect(last_patch_j, last_patch_i, patch_size, patch_size);
    Mat patch = image(patch_rect).clone();
    string name = "i" + to_string(last_patch_j) + "endi_j" + to_string(last_patch_i) + "endj";
    image_arr.push_back(patch);
    name_arr.push_back(name);
    count++;
    cout << "[SR7 INFO] Edge patches created";
    cout << "[SR7 INFO] Total patches created: " << count << endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage with img saving: ./image_patcher <image_path> <patch_size> <stride> <save> <output_folder>\n";
        cout << "Usage with w/ saving: ./image_patcher <image_path> <patch_size> <stride> \n";
        return -1;
    }

    string outputFolder;

    string imagePath = argv[1];
    int patchSize = stoi(argv[2]);
    float stride = stof(argv[3]);
    bool save = argv[4];
    if (save){
        outputFolder = argv[5];
    }
    cout << "[SR7 INFO] Loading image..." << endl;
    Mat image = imread(imagePath);
    if (image.empty()) {
        cout << "Error: Unable to load image from " << imagePath << endl;
        return -1;
    }

    vector<Mat> image_arr;
    vector<string> name_arr;

    if (save){
        patch_image(image, image_arr, name_arr, patchSize, stride, outputFolder);
    }
    else{
        patch_image(image, image_arr, name_arr, patchSize, stride);
    }
    
    return 0;
}