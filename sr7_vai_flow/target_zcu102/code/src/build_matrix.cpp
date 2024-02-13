#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

/*
def fill_with_1(grid, i, j ,endi, endj):
    temp = grid[i:endi, j:endj]
    ones = np.ones((endi-i,endj-j))
    grid[i:endi, j:endj] = temp + ones
*/

void fill_with_1(Mat& grid, int i, int j, int endi, int endj) {
    Mat temp(endi - i, endj - j, CV_8U, Scalar(0));
    for (int x = i, it = 0; x < endi && it < endi - i; ++x, ++it) {
        for (int y = j, jt = 0; y < endj && jt < endj - j; ++y, ++jt) {
            temp.at<uchar>(it, jt) = grid.at<uchar>(x, y); // Utiliser at<uchar>() pour accéder aux éléments de la matrice
        }
    }
    for (int x = i, it = 0; x < endi && it < endi - i; ++x, ++it) {
        for (int y = j, jt = 0; y < endj && jt < endj - j; ++y, ++jt) {
            grid.at<uchar>(x, y) = temp.at<uchar>(it, jt) + 1; // Utiliser at<uchar>() pour accéder aux éléments de la matrice
        }
    }
}
/*
def build_grid0(grid, patch_size, stride, line_width, color):
    '''
        Build the grid for regular patches starting from the top left corner of the image    
    '''
    for s_i in range(0, grid.shape[0], stride):
        for s_j in range(0, grid.shape[1], stride):
            if s_i+patch_size < grid.shape[0] and s_j+patch_size < grid.shape[1]:
                if s_i%(2*stride) == 0 and s_j%(2*stride) == 0:
                    fill_with_1(grid, s_i, s_j, s_i+patch_size, s_j+patch_size)
                    if s_i+stride+patch_size < grid.shape[0] and s_j+stride+patch_size < grid.shape[1]:
                        fill_with_1(grid, s_i+stride, s_j+stride, s_i+stride+patch_size, s_j+stride+patch_size)
    return grid
*/

void build_grid0(Mat& grid, int patch_size, int stride) {
    // Build the grid for regular patches starting from the top left corner of the image
    int height = grid.rows; // Utiliser rows pour obtenir la hauteur de la matrice
    int width = grid.cols; // Utiliser cols pour obtenir la largeur de la matrice

    for (int s_i = 0; s_i < height; s_i += stride) {
        for (int s_j = 0; s_j < width; s_j += stride) {
            if (s_i + patch_size < height && s_j + patch_size < width) {
                if (s_i % (2 * stride) == 0 && s_j % (2 * stride) == 0) {
                    fill_with_1(grid, s_i, s_j, s_i + patch_size, s_j + patch_size);
                    if (s_i + stride + patch_size < height && s_j + stride + patch_size < width) {
                        fill_with_1(grid, s_i + stride, s_j + stride, s_i + stride + patch_size, s_j + stride + patch_size);
                    }
                }
            }
        }
    }
}


void build_grid1(Mat& grid, int patch_size, int stride) {
    int height = grid.rows;
    int width = grid.cols;

    for (int s_i = 0; s_i < height; s_i += stride) {
        for (int s_j = 0; s_j < width; s_j += stride) {
            if (s_i + patch_size < height && s_j + patch_size < width) {
                if (s_i % (2 * stride) == 0 && s_j % (2 * stride) == 0) {
                    if (s_i == 0 && s_j + 2 * stride < width) {
                        fill_with_1(grid, s_i, s_j + stride, s_i + patch_size, s_j + patch_size + stride);
                    }
                    if (s_i + patch_size + stride < height && s_j + patch_size < width) {
                        fill_with_1(grid, s_i + stride, s_j, s_i + stride + patch_size, s_j + patch_size);
                    }
                    if (s_i + 2 * stride + patch_size < height && s_j + 2 * stride < width) {
                        fill_with_1(grid, s_i + 2 * stride, s_j + stride, s_i + 2 * stride + patch_size, s_j + stride + patch_size);
                    }}}}}}

void build_grid2(Mat& grid, int patch_size, int stride) {
    int height = grid.rows;
    int width = grid.cols;

    for (int s_i = 0; s_i < height; s_i += stride) {
        if (s_i + patch_size < height) {
            fill_with_1(grid, s_i, width - patch_size, s_i + patch_size, width);
        }
    }
    for (int s_j = 0; s_j < width; s_j += stride) {
        if (s_j + patch_size < width) {
            fill_with_1(grid, height - patch_size, s_j, height, s_j + patch_size);
        }
    }
    fill_with_1(grid, height - patch_size, width - patch_size, height, width);
}


int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./build_matrix <image_path>\n";
        return -1;
    }

    // Chemin d'accès à l'image
    String path_sensor_image = argv[1];
    cout << "[SR7 INFO] Loading the image from " << path_sensor_image << endl;
    Mat image = imread(path_sensor_image);
    cout << "[SR7 INFO] Image loaded" << endl;
    cout << "[SR7 INFO] Image shape: " << image.size() << endl;

    // Paramètres des patches
    int patch_size = 256;
    double stride_ratio = 0.1;
    int stride = static_cast<int>((1 - stride_ratio) * patch_size);
    int n_patches_i = image.rows / stride;
    int n_patches_j = image.cols / stride;
    int n_patches_edge = n_patches_i + n_patches_j + 1;
    int n_patches = n_patches_i * n_patches_j + n_patches_edge;

    Mat grid0(image.rows, image.cols, CV_8U, Scalar(0));
    Mat grid1(image.rows, image.cols, CV_8U, Scalar(0));
    Mat grid2(image.rows, image.cols, CV_8U, Scalar(0));

    build_grid0(grid0, patch_size, stride);
    build_grid1(grid1, patch_size, stride);
    build_grid2(grid2, patch_size, stride);

    Mat grid = grid0 + grid1 + grid2;

    //grid = 1 / grid * 255; //For display
    //cout << grid.rowRange(grid.rows - 100, grid.rows).colRange(grid.cols - 100, grid.cols) << endl;

    grid.convertTo(grid, CV_8U);
    imwrite("grid.png", grid);

    cout << "[SR7 INFO] Grid shape: " << image.size() << endl;
    cout << "[SR7 INFO] Patch size: " << patch_size << "x" << patch_size << endl;
    cout << "[SR7 INFO] Stride ratio: " << static_cast<int>(stride_ratio * 100) << "%" << endl;
    cout << "[SR7 INFO] Stride: " << stride << endl;
    cout << "[SR7 INFO] Number of patches: " << n_patches << endl;

    // cout << "[SR7 WARNING] Press 'q' to quit (or ctrl+C in the terminal)" << endl;
    // while (true) {
    //     char key = waitKey(1);
    //     if (key == 'q' || key == 'Q')
    //         break;
    // }
    // destroyAllWindows();

    return 0;
}

/*import cv2
import numpy as np

def fill_with_1(grid, i, j ,endi, endj):
    temp = grid[i:endi, j:endj]
    ones = np.ones((endi-i,endj-j))
    grid[i:endi, j:endj] = temp + ones

def build_grid0(grid, patch_size, stride, line_width, color):
    '''
        Build the grid for regular patches starting from the top left corner of the image    
    '''
    for s_i in range(0, grid.shape[0], stride):
        for s_j in range(0, grid.shape[1], stride):
            if s_i+patch_size < grid.shape[0] and s_j+patch_size < grid.shape[1]:
                if s_i%(2*stride) == 0 and s_j%(2*stride) == 0:
                    fill_with_1(grid, s_i, s_j, s_i+patch_size, s_j+patch_size)
                    if s_i+stride+patch_size < grid.shape[0] and s_j+stride+patch_size < grid.shape[1]:
                        fill_with_1(grid, s_i+stride, s_j+stride, s_i+stride+patch_size, s_j+stride+patch_size)
    return grid

def build_grid1(grid, patch_size, stride, line_width, color):
    '''
        Build the grid for complementary regular patches of grid0
    '''
    for s_i in range(0, grid.shape[0], stride):
        for s_j in range(0, grid.shape[1], stride):
            if s_i+patch_size < grid.shape[0] and s_j+patch_size < grid.shape[1]:
                if s_i%(2*stride) == 0 and s_j%(2*stride) == 0:
                    if s_i==0 and s_j+2*stride < grid.shape[1]:
                        fill_with_1(grid, s_i, s_j+stride, s_i+patch_size, s_j+patch_size+stride)
                    if s_i+patch_size+stride < grid.shape[0] and s_j+patch_size < grid.shape[1]:
                        fill_with_1(grid, s_i+stride, s_j, s_i+stride+patch_size, s_j+patch_size)
                    if s_i+2*stride+patch_size < grid.shape[0] and s_j+2*stride < grid.shape[1]:
                        fill_with_1(grid, s_i+2*stride, s_j+stride, s_i+2*stride+patch_size, s_j+stride+patch_size)
    return grid

def build_grid2(grid, patch_size, stride, line_width, color):
    '''
        Build the grid for irregular patches on the edges of the image
    '''
    for s_i in range(0, grid.shape[0], stride):
        if s_i+patch_size < grid.shape[0]:
            fill_with_1(grid, s_i, grid.shape[1]-patch_size, s_i+patch_size, grid.shape[1])
    for s_j in range(0, grid.shape[1], stride):
        if s_j+patch_size < grid.shape[1]:
            fill_with_1(grid, grid.shape[0]-patch_size, s_j, grid.shape[0], s_j+patch_size)
    fill_with_1(grid, grid.shape[0]-patch_size, grid.shape[1]-patch_size, grid.shape[0], grid.shape[1])
    return grid


def putGridOnImage(image, grid, color):
    '''
        Put the grid on the image by saturating the pixels of the image where the grid is not black
    '''
    vec_grid = grid.reshape(-1,3)
    vec_image = image.reshape(-1,3)
    idx = np.where(vec_grid[:,:] != 0)
    if color == "blue":
        vec_image[idx,0] = 255
    elif color == "green":
        vec_image[idx,1] = 255
    elif color == "red":
        vec_image[idx,2] = 255
    return vec_image.reshape(image.shape)


def displayGriddedImage(image, grid0, grid1, grid2):
    '''
        Display the image with the grid
    '''
    grid_image = putGridOnImage(image.copy(), grid0, "blue")
    grid_image = putGridOnImage(grid_image, grid1, "red")
    grid_image = putGridOnImage(grid_image, grid2, "green")

    x = 1000
    y = int(x*grid_image.shape[0]/grid_image.shape[1])
    grid_image = grid_image[-5*y:,-5*x:]

    small_grid_image = cv2.resize(grid_image, (x, y))
    del grid_image
    cv2.imshow("grid_image", small_grid_image)


# Load the PNG file
print("[SR7 INFO] Loading the file...")
path_sensor_image = "/home/eau_kipik/Images/2.png"
image = cv2.imread(path_sensor_image)
image = np.array(image)
print("[SR7 INFO] Image loaded")
print("[SR7 INFO] Image shape: ", image.shape)

# Patches settings
patch_size = 256
stride_ratio = 0.1
stride = int((1-stride_ratio)* patch_size)
n_patches_i = int(image.shape[0]//stride)
n_patches_j = int(image.shape[1]//stride)
n_patches_edge = n_patches_i+n_patches_j+1
n_patches = n_patches_i*n_patches_j + n_patches_edge

grid0 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
grid1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
grid2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
grid0 = build_grid0(grid0, patch_size, stride)
grid1 = build_grid1(grid1, patch_size, stride)
grid2 = build_grid2(grid2, patch_size, stride)
grid = grid0+grid1+grid2
grid = 1/grid
print(grid[-100:,-100:])
grid = (grid*255).astype(np.uint8)
print(grid)

print("[SR7 INFO] Image shape : ", image.shape)
print("[SR7 INFO] Patch size : ", patch_size,"x", patch_size)
print("[SR7 INFO] Stride ratio : ", int(stride_ratio*100), "%")
print("[SR7 INFO] Stride : ", stride)
print("[SR7 INFO] Number of patches : ", n_patches)
#del image, grid0, grid1, grid2

print("[SR7 WARNING] Press 'q' to quit (or ctrl+C in the terminal)")
while True:
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break
cv2.destroyAllWindows()*/