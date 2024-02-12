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
path_sensor_image = "1.png"
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

# Display settings
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
line_width = 4

grid0 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
grid1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
grid2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
grid0 = build_grid0(grid0, patch_size, stride, line_width, blue)
grid1 = build_grid1(grid1, patch_size, stride, line_width, red)
grid2 = build_grid2(grid2, patch_size, stride, line_width, green)
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
#displayGriddedImage(image, grid0, grid1, grid2)
#del image, grid0, grid1, grid2

# # --------------------------------------------------------------------
# # ----------------- Test the grid building functions -----------------
# # --------------------------------------------------------------------
# grid_image0 = putGridOnImage(image.copy(), grid0, "blue")
# grid_image1 = putGridOnImage(image.copy(), grid1, "red")
# grid_image2 = putGridOnImage(image.copy(), grid2, "green")
x = 1000
y = int(x*grid0.shape[0]/grid0.shape[1])
grid_image0 = cv2.resize(grid0, (x, y))
grid_image1 = cv2.resize(grid1, (x, y))
grid_image2 = cv2.resize(grid2, (x, y))
#grid_image = cv2.resize(grid, (x, y))
cv2.imshow("grid_image0", grid_image0)
cv2.imshow("grid_image1", grid_image1)
cv2.imshow("grid_image2", grid_image2)
cv2.imshow("grid_image", grid)

print("[SR7 WARNING] Press 'q' to quit (or ctrl+C in the terminal)")
while True:
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break
cv2.destroyAllWindows()*/