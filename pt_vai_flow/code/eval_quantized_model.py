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
import matplotlib.pyplot as plt

# import models and data loaders
from models import *
from data_loaders import *

# ==========================================================================================
# Get Input Arguments
# ==========================================================================================
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Vitis AI Pytorch Evaluation")

    # model type
    parser.add_argument("--model_type", type=str, default="fsrcnn",
                        help="model type")
    # model path
    parser.add_argument("--float_model_file", type=str,
                        help="pt floating point file path name")
    # quantization path
    parser.add_argument("--quantized_model_file", type=str,
                        help="pt quantized model file path name ")
    # dataset directory
    parser.add_argument("--dataset_dir", type=str, default="../supres_dataset",
                        help="dataset directory")
    # number of images to use for evaluation
    parser.add_argument("--eval_num_img", type=int, default=None,
                        help="number of images to use for evaluation")
    # batch size
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size")
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

    if args.save_images:
        SAVING_DIR = args.save_images_dir
        FLOAT_DIR = os.path.join(SAVING_DIR, "float")
        QUANT_DIR = os.path.join(SAVING_DIR, "quant")
        INPUT_DIR = os.path.join(SAVING_DIR, "input")
        LABEL_DIR = os.path.join(SAVING_DIR, "label")

    # ==========================================================================================
    # prepare your data
    # ==========================================================================================
    print("\n[SR7 INFO] Loading Test Data ...")
    
    if args.model_type == "fsrcnn":
        input_dir = os.path.join(args.dataset_dir, 'test/blr')
        label_dir = os.path.join(args.dataset_dir, 'test/gt')
        if args.eval_num_img is None:
            args.eval_num_img = len(os.listdir(input_dir))
        dataset = SuperResolutionDataset(input_dir, label_dir, args.eval_num_img, transform=transforms.ToTensor())
    elif args.model_type == "fcn8":
        input_dir = os.path.join(args.dataset_dir, 'train/image')
        label_dir = os.path.join(args.dataset_dir, 'train/mask')
        dataset = SegmentationDataset(input_dir, label_dir, args.eval_num_img, ratio=1, patch_size=256)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    X_test_shape = np.insert(np.array(dataset[0][0].shape), 0, len(dataset))
    Y_test_shape = np.insert(np.array(dataset[0][1].shape), 0, len(dataset))
    print("--------> X_test shape = ", X_test_shape)
    print("--------> Y_test shape = ", Y_test_shape)

    # Get the input shape from the first data of the data_loader
    for input_data, label_data in data_loader:
        input_shape = input_data.shape[1:]
        output_shape = label_data.shape[1:]
        break
    print("[SR7 INFO] Input shape: ", input_shape, ". Output shape: ", output_shape)

    # Initialize the model
    if args.model_type == "fsrcnn":
        upscale_factor = 2
        color_channels = input_shape[0]
        model = FSRCNN(d=56, s=16, m=6, input_size=input_shape,   upscaling_factor=upscale_factor, color_channels=color_channels)
        criterion = nn.MSELoss()
    elif args.model_type == "fcn8":
        num_classes = output_shape[0]
        model = FCN8(nClasses=num_classes)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Model type not found")
    
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
        for input_data, label_data in data_loader:
            input_images, output_label = input_data.to(device), label_data.to(device)
            output_predict = model(input_images)
            test_results.append(criterion(output_predict, output_label.float()).item())
    test_results = np.array(test_results)
    test_results = np.mean(test_results)
    if args.model_type == "fsrcnn":
        print("--------> Results on Test Dataset with Float Model:", PSNR(test_results))
    elif args.model_type == "fcn8":
        print("--------> Results on Test Dataset with Float Model:", test_results)

    ## Quantized Model
    print("[SR7 INFO] Evaluation of Quantized Model...")
    q_model.eval()
    q_test_results = []
    with torch.no_grad():
        for input_data, label_data in data_loader:
            input_images, output_label = input_data.to(device), label_data.to(device)
            output_predict = q_model(input_images)
            q_test_results.append(criterion(output_predict, output_label.float()).item())
    q_test_results = np.array(q_test_results)
    q_test_results = np.mean(q_test_results)
    if args.model_type == "fsrcnn":
        print("--------> Results on Test Dataset with Float Model:", PSNR(q_test_results))
        print("--------> Drop: ", PSNR(test_results) - PSNR(q_test_results))
    elif args.model_type == "fcn8":
        print("--------> Results on Test Dataset with Float Model:", q_test_results)
        print("--------> Drop: ", test_results - q_test_results)
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
            for input_data, label_data in data_loader:
                input_images, output_label = input_data.to(device), label_data.to(device)
                output_predict = model(input_images)
                Y_pred_float.append(output_predict)
                output_predict_q = q_model(input_images)
                Y_pred_q.append(output_predict_q)        
                X_test.append(input_images)
                Y_test.append(output_label)
        
        torch.save(Y_pred_float, os.path.join(SAVING_DIR, "Y_pred_float.pt"))
        torch.save(Y_pred_q, os.path.join(SAVING_DIR, "Y_pred_q.pt"))
        torch.save(X_test, os.path.join(SAVING_DIR, "X_test.pt"))
        torch.save(Y_test, os.path.join(SAVING_DIR, "Y_test.pt"))


        print("[SR7 INFO] Saving Images...")
        # Save in png format
        for i in range(len(Y_pred_float)):
            if args.model_type == "fcn8":
                Y_pred_mask = dataset.masks_2_RGB(Y_pred_float[i][0]).numpy()
                Y_pred_q_mask = dataset.masks_2_RGB(Y_pred_q[i][0]).numpy()
                Y_test_mask = dataset.masks_2_RGB(Y_test[i][0]).numpy()
                
            cv2.imwrite(os.path.join(FLOAT_DIR, f"float_{i}.png"), cv2.cvtColor(Y_pred_mask, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(QUANT_DIR, f"quant_{i}.png"), cv2.cvtColor(Y_pred_q_mask, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(LABEL_DIR, f"label_{i}.png"), cv2.cvtColor(Y_test_mask, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(INPUT_DIR, f"input_{i}.png"), cv2.cvtColor(X_test[i][0].cpu().numpy().transpose(1, 2, 0) * 255, cv2.COLOR_BGR2RGB))

        print("[SR7 INFO] Images saved in ", SAVING_DIR)
# ==========================================================================================
        
print("[SR7 INFO] Evaluation done!\n")

# ==========================================================================================
# END
# ==========================================================================================

if __name__ == '__main__':
    main()
