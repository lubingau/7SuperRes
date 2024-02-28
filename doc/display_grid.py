import cv2
import numpy as np


def build_grid0(grid, patch_size, stride, line_width, color):
    '''
        Build the grid for regular patches starting from the top left corner of the image    
    '''
    overlap = patch_size - stride
    for s_i in range(0, grid.shape[0], stride):
        for s_j in range(0, grid.shape[1], stride):
            if s_i+patch_size < grid.shape[0]-overlap and s_j+patch_size < grid.shape[1]-overlap:
                if s_i%(2*stride) == 0 and s_j%(2*stride) == 0:
                    cv2.rectangle(grid, (s_j, s_i), (s_j+patch_size, s_i+patch_size), color, line_width)
                    if s_i+stride+patch_size < grid.shape[0]-overlap and s_j+stride+patch_size < grid.shape[1]-overlap:
                        cv2.rectangle(grid, (s_j+stride, s_i+stride), (s_j+stride+patch_size, s_i+stride+patch_size), color, line_width)
    return grid

def build_grid1(grid, patch_size, stride, line_width, color):
    '''
        Build the grid for complementary regular patches of grid0
    '''
    overlap = patch_size - stride
    for s_i in range(0, grid.shape[0], stride):
        for s_j in range(0, grid.shape[1], stride):
            if s_i+patch_size+stride < grid.shape[0]:
                if s_i%(2*stride) == 0 and s_j%(2*stride) == 0:
                    if s_j+patch_size < grid.shape[1]-patch_size:
                        cv2.rectangle(grid, (s_j+stride, s_i), (s_j+stride+patch_size, s_i+patch_size), color, line_width)
                    if s_i+patch_size < grid.shape[0]-patch_size:
                        cv2.rectangle(grid, (s_j, s_i+stride), (s_j+patch_size, s_i+stride+patch_size), color, line_width)
    return grid

def build_grid2(grid, patch_size, stride, line_width, color):
    '''
        Build the grid for irregular patches on the edges of the image
    '''
    overlap = patch_size - stride
    for s_i in range(0, grid.shape[0], stride):
        if s_i+patch_size < grid.shape[0]-overlap:
            cv2.rectangle(grid, (grid.shape[1], s_i), (grid.shape[1]-patch_size, s_i+patch_size), color, line_width)
    for s_j in range(0, grid.shape[1], stride):
        if s_j+patch_size < grid.shape[1]-overlap:
            cv2.rectangle(grid, (s_j, grid.shape[0]), (s_j+patch_size, grid.shape[0]-patch_size), color, line_width)
    cv2.rectangle(grid, (grid.shape[1], grid.shape[0]), (grid.shape[1]-patch_size, grid.shape[0]-patch_size), color, line_width)
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

    # small_grid_image = cv2.resize(grid_image, (x, y),interpolation = cv2.INTER_AREA)
    small_grid_image = np.array(grid_image)
    del grid_image
    cv2.imwrite("grid_image.png", small_grid_image)
    # cv2.imshow("grid_image", small_grid_image)


# Load the PNG file
print("[SR7 INFO] Loading the file...")
path_sensor_image = "resized_2.png"
image = cv2.imread(path_sensor_image)
image = np.array(image)
print("[SR7 INFO] Image loaded")
print("[SR7 INFO] Image shape: ", image.shape)

# Patches settings
patch_size = 128
stride_ratio = 0.9
stride = int(patch_size*stride_ratio)
n_patches_i = int(image.shape[0]//stride)
n_patches_j = int(image.shape[1]//stride)
n_patches_edge = n_patches_i+n_patches_j+1
n_patches = n_patches_i*n_patches_j + n_patches_edge

# Display settings
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
line_width = 1

grid0 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
grid1 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
grid2 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

grid0 = build_grid0(grid0, patch_size, stride, line_width, blue)
grid1 = build_grid1(grid1, patch_size, stride, line_width, red)
grid2 = build_grid2(grid2, patch_size, stride, line_width, green)

print("[SR7 INFO] Image shape : ", image.shape)
print("[SR7 INFO] Patch size : ", patch_size,"x", patch_size)
print("[SR7 INFO] Stride ratio : ", int(stride_ratio*100), "%")
print("[SR7 INFO] Stride : ", stride)
print("[SR7 INFO] Number of patches : ", n_patches)
displayGriddedImage(image, grid0, grid1, grid2)
del image, grid0, grid1, grid2

# # --------------------------------------------------------------------
# # ----------------- Test the grid building functions -----------------
# # --------------------------------------------------------------------
# grid_image0 = putGridOnImage(image.copy(), grid0, "blue")
# grid_image1 = putGridOnImage(image.copy(), grid1, "red")
# grid_image2 = putGridOnImage(image.copy(), grid2, "green")
# cv2.imshow("grid_image0", grid_image0)
# cv2.imshow("grid_image1", grid_image1)
# cv2.imshow("grid_image2", grid_image2)

print("[SR7 WARNING] Press 'q' to quit (or ctrl+C in the terminal)")
while True:
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break
cv2.destroyAllWindows()