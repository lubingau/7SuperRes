#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# ==========================================================================================
# import dependencies
# ==========================================================================================


from tensorflow import keras
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from config import config as cfg


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
    # number of images to use for evaluation
    parser.add_argument("--eval_num_img", type=int, default=None,
                        help="number of images to use for evaluation")
    # save predicted images
    parser.add_argument("--save_images", action="store_true",
                        help="save predicted images")
    # saving path
    parser.add_argument("--save_images_dir", type=str,
                        help="saving directory")
    
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
SAVING_DIR = args.save_images_dir

FLOAT_DIR = os.path.join(SAVING_DIR, "float")
QUANT_DIR = os.path.join(SAVING_DIR, "quant")
GT_DIR = os.path.join(SAVING_DIR, "gt")
LR_DIR = os.path.join(SAVING_DIR, "blr")

# ==========================================================================================
# prepare your data
# ==========================================================================================
print("\n[SR7 INFO] Loading Test Data ...")

dir_test_input = cfg.dir_test_input
dir_test_label = cfg.dir_test_label

if args.eval_num_img is None:
    max_images = len(os.listdir(dir_test_input))
else:
    max_images = args.eval_num_img

# Load data from a folder  
def load_data(dir, max_images):
    X_data = []
    filelist = os.listdir(dir)
    filelist = sorted(filelist)
    np.random.seed(0)
    np.random.shuffle(filelist)
    filelist = filelist[:max_images]
    for filename in filelist:
        img = plt.imread(os.path.join(dir, filename))
        X_data.append(img)
    return np.array(X_data)

X_test = load_data(dir_test_input, max_images)
Y_test = load_data(dir_test_label, max_images)

print("--------> X_test shape = ", X_test.shape)
print("--------> Y_test shape = ", Y_test.shape)

# ==========================================================================================
# Load Float and Quantized Models
# ==========================================================================================
print("[SR7 INFO] Loading Float Model...")
model = keras.models.load_model(FLOAT_H5_FILE)

print("[SR7 INFO] Loading Quantized Model...")
q_model = keras.models.load_model(QUANT_H5_FILE)

# ==========================================================================================
# Evaluations
# ==========================================================================================
## Float Model
print("[SR7 INFO] Evaluation with Float Model...")
test_results = model.evaluate(X_test, Y_test)
if isinstance(test_results, list):
    test_results = test_results[0]
print("--------> Results on Test Dataset with Float Model:", PSNR(test_results))

## Quantized Model
print("[SR7 INFO] Evaluation of Quantized Model...")
with vitis_quantize.quantize_scope():
    q_model.compile(optimizer="rmsprop", loss="mse")
    q_eval_results = q_model.evaluate(X_test, Y_test)
    print("--------> Results on Test Dataset with Quantized Model: ", PSNR(q_eval_results))
    print("--------> Drop: ", PSNR(test_results) - PSNR(q_eval_results))
# ==========================================================================================

if args.save_images: 
    # ==========================================================================================
    # Saving images
    # ==========================================================================================

    # Predictions of floating point model
    print("[SR7 INFO] Predictions of Floating Point Model...")
    Y_pred_float = model.predict(X_test)
    Y_pred_float = Y_pred_float - np.min(Y_pred_float)
    Y_pred_float = Y_pred_float / np.max(Y_pred_float)

    # Predictions of quantized model
    print("[SR7 INFO] Predictions of Quantized Model...")
    Y_pred = q_model.predict(X_test)
    Y_pred = Y_pred - np.min(Y_pred)
    Y_pred = Y_pred / np.max(Y_pred)

    print("[SR7 INFO] Saving Images...")
    # Save in png format
    for i in range(len(Y_pred)):
        plt.imsave(os.path.join(FLOAT_DIR, "float_" + str(i) + ".png"), Y_pred_float[i])
        plt.imsave(os.path.join(QUANT_DIR, "quant_" + str(i) + ".png"), Y_pred[i])
        plt.imsave(os.path.join(GT_DIR, "gt_" + str(i) + ".png"), Y_test[i])
        plt.imsave(os.path.join(LR_DIR, "blr_" + str(i) + ".png"), X_test[i])

# ==========================================================================================
        
print("[SR7 INFO] Evaluation done!\n")

# ==========================================================================================
# END
# ==========================================================================================

