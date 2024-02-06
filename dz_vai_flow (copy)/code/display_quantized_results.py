import matlab
import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import cv2
import random
import matplotlib.pyplot as plt
import mplcursors


def PSNR(mse):
    """Calculate PSNR from MSE."""
    return - 10.0 * np.log10(1.0 / mse)

global_dir = "../build/2_predictions/fsrcnn6/"
gt_dir = "gt/"
lr_dir = "lr/"
float_dir = "float/"
quant_dir = "quant/"

N = 1000
n = 6
random.seed(1)
randomlist = random.sample(range(0, N), n)

general_size = 1
fontsize = 8*general_size
fig = plt.figure(figsize=(general_size*18, general_size*9))
mplcursors.cursor(hover=True)

for i in range(len(randomlist)):
    
    im_gt = cv2.imread(global_dir + gt_dir + "gt_" + str(randomlist[i]) + ".png")
    im_lr = cv2.imread(global_dir + lr_dir + "lr_" + str(randomlist[i]) + ".png")
    im_float = cv2.imread(global_dir + float_dir + "float_" + str(randomlist[i]) + ".png")
    im_quant = cv2.imread(global_dir + quant_dir + "quant_" + str(randomlist[i]) + ".png")

    psnr_float = PSNR(np.mean((im_gt.astype(np.float32) - im_float.astype(np.float32)) ** 2))
    psnr_quant = PSNR(np.mean((im_gt.astype(np.float32) - im_quant.astype(np.float32)) ** 2))

    ax = plt.subplot(4,n,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im_gt.astype(np.uint8))
    ax.set_title("GT", fontsize=fontsize)
    ax.set_ylabel(f"{im_gt.shape[0]}x{im_gt.shape[1]}", fontsize=fontsize)

    ax = plt.subplot(4,n,i+1+n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im_lr.astype(np.uint8))
    ax.set_title("Input(LR)", fontsize=fontsize)
    ax.set_ylabel(f"{im_lr.shape[0]}x{im_lr.shape[1]}", fontsize=fontsize)

    ax = plt.subplot(4,n,i+1+2*n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im_float.astype(np.uint8))
    ax.set_title("FSRCNN-6 Float", fontsize=fontsize)
    # ax.set_xlabel(f"PSNR: {psnr_float:.2f}", fontsize=fontsize)
    ax.set_ylabel(f"{im_float.shape[0]}x{im_float.shape[1]}", fontsize=fontsize)

    ax = plt.subplot(4,n,i+1+3*n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im_quant.astype(np.uint8))
    ax.set_title("FSRCNN-6 Quantized", fontsize=fontsize)
    # ax.set_xlabel(f"PSNR: {psnr_quant:.2f}", fontsize=fontsize)
    ax.set_ylabel(f"{im_quant.shape[0]}x{im_quant.shape[1]}", fontsize=fontsize)

plt.savefig('quantized_results.png')
plt.show()
