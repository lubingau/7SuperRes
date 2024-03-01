#!/usr/bin/env python

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms

from torchsummary import summary
import numpy as np
import cv2
from PIL import Image
import scipy.io as scio
from sr_model import FSRCNN
from data_loader import SuperResolutionDataset
import matplotlib.pyplot as plt

# ==========================================================================================
# Get Input Arguments
# ==========================================================================================
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Vitis AI Pytorch Evaluation")

    # model config
    parser.add_argument("--float_model_file", type=str,
                        help="pt floating point file path name")
    # quantization config
    parser.add_argument("--quantized_model_file", type=str,
                        help="pt quantized model file path name ")
    # dataset directory
    parser.add_argument("--dataset_dir", type=str, default="../../supres_dataset",
                        help="dataset directory")
    # number of images to use for evaluation
    parser.add_argument("--eval_num_img", type=int, default=None,
                        help="number of images to use for evaluation")
    # save predicted images
    parser.add_argument("--save_images", action="store_true",
                        help="save predicted images")
    # saving path
    parser.add_argument("--save_images_dir", type=str,
                        help="saving directory")
    # device
    parser.add_argument("--device", default="cpu", type=str)
    
    ## IF YOU WANT TO USE GPU UNCOMMENT THE FOLLOWING LINE
    # parser.add_argument("--gpus", type=str, default="0",
    #                     help="choose gpu devices.")

    return parser.parse_args()

def PSNR(mse):
    """Calculate PSNR from MSE."""
    return 10.0 * np.log10(1.0 / mse)


def main():
    args = get_arguments()

    torch.manual_seed(0)
    if (args.device == 'gpu') and (torch.cuda.is_available()) :
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Testing on {device} device.")

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

    transform = transforms.Compose([transforms.ToTensor()])

    blr_dir = os.path.join(args.dataset_dir, 'test/blr')
    gt_dir = os.path.join(args.dataset_dir, 'test/gt')
    if args.eval_num_img is None:
        args.eval_num_img = len(os.listdir(blr_dir))
    
    super_resolution_dataset = SuperResolutionDataset(blr_dir, gt_dir, args.eval_num_img, transform)

    super_resolution_loader = torch.utils.data.DataLoader(
        super_resolution_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=4
    )
    print("--------> X_test shape = ", super_resolution_dataset[0][0].shape)
    print("--------> Y_test shape = ", super_resolution_dataset[0][1].shape)

    # Get the input shape from the first data of the super_resolution_loader
    for blr_data, gt_data in super_resolution_loader:
        input_shape = blr_data.shape[1:]
        break
    print(input_shape)
    upscale_factor = 2
    color_channels = input_shape[0]

    # Initialize the model
    model = FSRCNN(d=56, s=16, m=6, input_size=input_shape,   upscaling_factor=upscale_factor, color_channels=color_channels)
    
    # ==========================================================================================
    # Load Float and Quantized Models
    # ==========================================================================================
    print("[SR7 INFO] Loading Float Model...")
    model.load_state_dict(torch.load(args.float_model_file, map_location=device), strict=True)
    summary(model, input_shape)

    print("[SR7 INFO] Loading Quantized Model...")
    import pytorch_nndct
    q_model = torch.jit.load(args.quantized_model_file)
    # summary(q_model, input_shape)

    # ==========================================================================================
    # Evaluations
    # ==========================================================================================
    ## Float Model
    print("[SR7 INFO] Evaluation with Float Model...")
    model.eval()
    test_results = []
    with torch.no_grad():
        for blr_data, gt_data in super_resolution_loader:
            blr_images, gt_images = blr_data.to(device), gt_data.to(device)
            sr_images = model(blr_images)
            # print(sr_images.shape)
            test_results.append(F.mse_loss(sr_images, gt_images).item())
    test_results = np.array(test_results)
    test_results = np.mean(test_results)
    print("--------> Results on Test Dataset with Float Model:", PSNR(test_results))

    ## Quantized Model
    print("[SR7 INFO] Evaluation of Quantized Model...")
    q_model.eval()
    q_test_results = []
    with torch.no_grad():
        for blr_data, gt_data in super_resolution_loader:
            blr_images, gt_images = blr_data.to(device), gt_data.to(device)
            sr_images = q_model(blr_images)
            q_test_results.append(F.mse_loss(sr_images, gt_images).item())
    q_test_results = np.array(test_results)
    q_test_results = np.mean(test_results)
    print("--------> Results on Test Dataset with Quantized Model:", PSNR(q_test_results))
    print("--------> Drop: ", PSNR(test_results) - PSNR(q_test_results))
    # ==========================================================================================

    if args.save_images:
        # ==========================================================================================
        # Saving images
        # ==========================================================================================

        # Predictions of floating point model
        print("[SR7 INFO] Predictions of Floating Point and Quantized Model...")
        model.eval()
        Y_pred_float = []
        q_model.eval()
        Y_pred_q = []
        X_test = []
        Y_test = []
        with torch.no_grad():
            for blr_data, gt_data in super_resolution_loader:
                blr_images, gt_images = blr_data.to(device), gt_data.to(device)
                sr_images = model(blr_images)
                Y_pred_float.append(sr_images)
                sr_images = q_model(blr_images)
                Y_pred_q.append(sr_images)        
                X_test.append(blr_images)
                Y_test.append(gt_images)


        print("[SR7 INFO] Saving Images...")
        # Save in png format
        for i in range(len(Y_pred_float)):
            cv2.imwrite(os.path.join(FLOAT_DIR, f"float_{i}.png"), Y_pred_float[i].squeeze().cpu().numpy().transpose(1, 2, 0) * 255)
            cv2.imwrite(os.path.join(QUANT_DIR, f"quant_{i}.png"), Y_pred_q[i].squeeze().cpu().numpy().transpose(1, 2, 0) * 255)
            cv2.imwrite(os.path.join(GT_DIR, f"gt_{i}.png"), Y_test[i].squeeze().cpu().numpy().transpose(1, 2, 0) * 255)
            cv2.imwrite(os.path.join(LR_DIR, f"lr_{i}.png"), X_test[i].squeeze().cpu().numpy().transpose(1, 2, 0) * 255)

        print("[SR7 INFO] Images saved in ", SAVING_DIR)
# ==========================================================================================
        
print("[SR7 INFO] Evaluation done!\n")

# ==========================================================================================
# END
# ==========================================================================================

if __name__ == '__main__':
    main()
