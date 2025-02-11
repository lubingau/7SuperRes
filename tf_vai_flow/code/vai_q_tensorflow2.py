#!/usr/bin/env python

from tensorflow import keras
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================================================
# Get Input Arguments
# ==========================================================================================
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Vitis AI TF2 Quantization of ResNet18 trained on CIFAR10")

    # model config
    parser.add_argument("--float_model_file", type=str,
                        help="h5 floating point file path name")
    # quantization config
    parser.add_argument("--quantized_model_file", type=str,
                        help="quantized model file path name ")
    # calibration iterations
    parser.add_argument("--calib_iter", type=int, default=10,
                        help="number of calibration iterations")
    # train images directory
    parser.add_argument("--train_images_dir", type=str, default="../../supres_dataset/train",
                        help="train images directory")
    # test images directory
    parser.add_argument("--test_images_dir", type=str, default="../../supres_dataset/test",
                        help="test images directory")
    # number of images to use for calibration
    parser.add_argument("--calib_num_img", type=int, default=None,
                        help="number of images to use for calibration")
    
    ## IF YOU WANT TO USE GPU UNCOMMENT THE FOLLOWING LINE
    # parser.add_argument("--gpus", type=str, default="0",
    #                     help="choose gpu devices.")

    return parser.parse_args()

def PSNR(mse):
    """Calculate PSNR from MSE."""
    return 10.0 * np.log10(1.0 / mse)

args = get_arguments()

# ==========================================================================================
# Global Variables
# ==========================================================================================

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

FLOAT_H5_FILE = args.float_model_file
QUANT_H5_FILE = args.quantized_model_file

# ==========================================================================================
# prepare your data
# ==========================================================================================
print("\n[SR7 INFO] Loading Train Data ...")

dir_train_input = os.path.join(args.train_images_dir, "blr")

if args.calib_num_img is None:
    max_images = len(os.listdir(dir_train_input))
else:
    max_images = args.calib_num_img

# Load data from a folder  
def load_data(dir, max_images):
    X_data = []
    filelist = os.listdir(dir)
    np.random.seed(0)
    np.random.shuffle(filelist)
    filelist = filelist[:max_images]
    for filename in filelist:
        img = plt.imread(os.path.join(dir, filename))
        X_data.append(img)
    return np.array(X_data)

X_train = load_data(dir_train_input, max_images)

print("--------> X_train shape = ", X_train.shape)

# ==========================================================================================
# Get the trained floating point model
# ==========================================================================================
print("[SR7 INFO] Loading Float Model...")

model = keras.models.load_model(FLOAT_H5_FILE)

# ==========================================================================================
# Vitis AI Quantization
# ==========================================================================================
print("[SR7 INFO] Vitis AI Quantization...")

from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)
q_model = quantizer.quantize_model(calib_dataset=X_train, calib_steps=args.calib_iter)

q_model.save(QUANT_H5_FILE)
print("[SR7 INFO] Saved Quantized Model in :", QUANT_H5_FILE)

print("[SR7 INFO] Quantization done!")

# ==========================================================================================
# END
# ==========================================================================================
