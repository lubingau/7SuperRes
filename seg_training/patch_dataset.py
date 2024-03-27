import os
import cv2
from multiprocessing import Pool
import numpy as np
from tqdm.auto import tqdm

import argparse

"""
Usecase example: 

python create_dataset.py --input_dir /path/to/input_images --output_dir dz_dataset/train/ --kernel_path blur_kernel.csv --stride 0.75 --patch_size 256 --nb_img -1

"""

def patch_image(img_path, mask_path, gt_out_folder, mask_out_folder, patch_size):
    
    """
    Extracts patches from an input image, applies a blur, and saves the patches to separate directories.

    Args:
        img_path (str): Path to the input image.
        gt_out_folder (str): Path to the directory to save high-resolution patches.
        mask_out_folder (str): Path to the directory to save blurred patches.
        patch_size (int): Size of the patches to extract.
    Returns:
        None
    """

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    height, width, _ = img.shape
    
    for i in range(0, height-patch_size, patch_size):
        
        for j in range(0, width-patch_size, patch_size):
            
            gt_patch = img[i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]
            
            img_name = os.path.basename(img_path)
            
            gt_patch_path = os.path.join(gt_out_folder, f"gt_{img_name[:-4]}_{i}_{j}.png")
            mask_patch_path = os.path.join(mask_out_folder,
                                           f"mask_{img_name[:-4]}_{i}_{j}.png")
            
            cv2.imwrite(gt_patch_path, gt_patch)
            cv2.imwrite(mask_patch_path, mask_patch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str, help='Path to input images directory')
    parser.add_argument('--output_dir', default='patches/train/', type=str, help='Path to save the dataset')
    parser.add_argument('--patch_size', default=256, type=int, help='Size of the patches to extract')
    parser.add_argument('--nb_img', default=-1, type=int, help='Number of images to process, -1 for full dataset')
    
    args = parser.parse_args()

    input_folder = args.input_dir
    gt_output_images_folder = os.path.join(args.output_dir, "gt")
    mask_output_images_folder = os.path.join(args.output_dir, "mask")
    patch_size = args.patch_size
    nb_img = args.nb_img
    
    os.makedirs(gt_output_images_folder, exist_ok=True)
    os.makedirs(mask_output_images_folder, exist_ok=True)
    
    images_dir = os.path.join(input_folder, 'images')
    masks_dir = os.path.join(input_folder, 'masks')
    
    images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.jpg', '.png'))])
    
    if nb_img == -1:
        nb_img = len(images)       

    with Pool() as p:
        p.starmap(patch_image, [(os.path.join(images_dir, img_name), os.path.join(masks_dir, mask_name), 
                                 gt_output_images_folder,
                                 mask_output_images_folder, patch_size) for img_name, mask_name in zip(images[:nb_img], masks[:nb_img])])