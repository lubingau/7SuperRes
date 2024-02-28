import cv2
import numpy as np

def buildTestImage(image, sensor_size):
    '''
        Build the test image to match the size of the sensor
        The input image is mirrored on both axes to fill the sensor
    '''
    sensor_image = np.zeros((sensor_size[0], sensor_size[1], 3), dtype=np.uint8)
    image_miror = image.copy()
    if image.shape[0] > sensor_size[0] or image.shape[1] > sensor_size[1]:
        print("[SR7 ERROR] The image is too big for the target sensor size")
        return None
    print("[SR7 INFO] Building sensor image...")
    for i in range(0, sensor_size[0], image.shape[0]):
        for j in range(0, sensor_size[1], image.shape[1]):
            if i+image.shape[0] < sensor_size[0] and j+image.shape[1] < sensor_size[1]:
                sensor_image[i:i+image.shape[0], j:j+image.shape[1]] = image_miror
                #cv2.rectangle(sensor_image, (j, i), (j+image.shape[1], i+image.shape[0]), (255,0,0), 100) # draw rectangles to view the construction of the image
            else:
                sensor_image[i:i+image.shape[0], j:j+image.shape[1]] = image_miror[:sensor_size[0]-i, :sensor_size[1]-j]
                #cv2.rectangle(sensor_image, (j, i), (j+image.shape[1], i+image.shape[0]), (255,0,0), 100) # draw rectangles to view the construction of the image
            image_miror = np.flip(image_miror, axis=1)
        image_miror = np.flip(image_miror, axis=0)
    print("[SR7 INFO] Sensor image built")
    return sensor_image

# Load the file
print("[SR7 INFO] Loading the file...")
image = cv2.imread("15.png") 
image = np.array(image)
print("[SR7 INFO] Original image shape: ", image.shape)

sensor_size = (10640, 14192)
target_dim = (sensor_size[0]*2, sensor_size[1]*2, 3)
print("[SR7 INFO] Target image shape: ", target_dim)
path_sensor_image = "sensor_image.png"

sensor_image = buildTestImage(image, target_dim)
print("[SR7 INFO] Sensor image shape: ", sensor_image.shape)

x = 800
y = int(x*sensor_size[0]/sensor_size[1])

small_sensor_image = cv2.resize(sensor_image, (x, y))
print("[SR7 INFO] Display image shape : ", small_sensor_image.shape)
#cv2.imshow("sensor_image", small_sensor_image)
cv2.imwrite(path_sensor_image, sensor_image)

# --------------------------------------------------------------------
# ----------------- Patch the sensor sized image -----------------
# --------------------------------------------------------------------
from create_dataset import patch_image

kernel = np.genfromtxt("PSF_E10x2.csv", delimiter=';')
kernel = kernel[:,:-1] # remove the last column of nan because of the delimiter

print("[SR7 INFO] Patching sensor image...")
patch_image(path_sensor_image, "test_image/gt", "test_image/blr", 256, kernel, 0.9)
print("[SR7 INFO] Sensor image patched")

# print("[SR7 WARNING] Press 'q' to quit (or ctrl+C in the terminal)")
# while True:
#     key = cv2.waitKey(1)
#     if key == ord('q') or key == ord('Q'):
#         break
# cv2.destroyAllWindows()