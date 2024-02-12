#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void save_patch(const Mat& patch, const string& folder, int row, int col) {
    string filename = folder + "/patch_i" + to_string(row) + "endi_j" + to_string(col) + "endj.png";
    imwrite(filename, patch);
}

void patch_image(const Mat& image, int patch_size, float stride, const string& folder) {
    int stride_pixels = static_cast<int>(stride * patch_size);
    int overlap = patch_size - stride_pixels;

    for (int y = 0; y < image.rows - stride_pixels; y += stride_pixels) {
        for (int x = 0; x < image.cols - stride_pixels; x += stride_pixels) {
            Rect patch_rect(x, y, patch_size, patch_size);
            Mat patch = image(patch_rect).clone();
            save_patch(patch, folder, y, x);
        }
    }

    int last_patch_x = image.cols - patch_size;
    int last_patch_y = image.rows - patch_size;

    for (int y = 0; y < image.rows - patch_size; y += stride_pixels) {
        Rect patch_rect(last_patch_x, y, patch_size, patch_size);
        Mat patch = image(patch_rect).clone();
        save_patch(patch, folder, y, last_patch_x);
    }

    for (int x = 0; x < image.cols - patch_size; x += stride_pixels) {
        Rect patch_rect(x, last_patch_y, patch_size, patch_size);
        Mat patch = image(patch_rect).clone();
        save_patch(patch, folder, last_patch_y, x);
    }

    Rect last_patch(last_patch_x, last_patch_y, patch_size, patch_size);
    Mat patch = image(last_patch).clone();
    save_patch(patch, folder, last_patch_y, last_patch_x);

}

int main(int argc, char** argv) {
    if (argc != 5) {
        cout << "Usage: ./image_patcher <image_path> <patch_size> <stride> <output_folder>\n";
        return -1;
    }

    string imagePath = argv[1];
    int patchSize = stoi(argv[2]);
    float stride = stof(argv[3]);
    string outputFolder = argv[4];

    Mat image = imread(imagePath);
    if (image.empty()) {
        cout << "Error: Unable to load image from " << imagePath << endl;
        return -1;
    }

    patch_image(image, patchSize, stride, outputFolder);
    return 0;
}