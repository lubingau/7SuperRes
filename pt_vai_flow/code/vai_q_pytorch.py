#!/usr/bin/env python

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from pytorch_nndct.apis import torch_quantizer
from torchsummary import summary

import numpy as np
import cv2
from PIL import Image
import scipy.io as scio
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
    parser = argparse.ArgumentParser(description="Vitis AI Pytorch Quantization")

    # model type
    parser.add_argument("--model_type", type=str, default="fsrcnn",
                        help="model type")
    # model config
    parser.add_argument("--float_model_file", type=str,
                        help="pt floating point file path name")
    # quantization config
    parser.add_argument("--quantized_model_dir", type=str,
                        help="quantized model file path name ")
    # dataset directory
    parser.add_argument("--dataset_dir", type=str, default="../supres_dataset",
                        help="dataset directory")
    # test batch size
    parser.add_argument("--test_batch_size", type=int, default=64,
                        help="test batch size")
    # quantization mode
    parser.add_argument("--quant_mode", default="calib", type=str,
                        help="quantization mode")
    # number of images to use for calibration
    parser.add_argument("--calib_num_img", type=int, default=None,
                        help="number of images to use for calibration")
    # deployment
    parser.add_argument("--deploy", action='store_true',
                        help="export model for deployment")
    # device
    parser.add_argument("--device", default="cpu", type=str)
    
    ## IF YOU WANT TO USE GPU UNCOMMENT THE FOLLOWING LINE
    # parser.add_argument("--gpus", type=str, default="0",
    #                     help="choose gpu devices.")

    return parser.parse_args()

def PSNR(mse):
    """Calculate PSNR from MSE."""
    return 10.0 * np.log10(1.0 / mse)

def test(model_type, model, device, test_loader, criterion):
    model.eval()
    q_test_results = []
    with torch.no_grad():
        for input_data, label_data in test_loader:
            input_images, output_label = input_data.to(device), label_data.to(device)
            output_predict = model(input_images)
            q_test_results.append(criterion(output_predict, output_label.float()).item())
    q_test_results = np.array(q_test_results)
    q_test_results = np.mean(q_test_results)
    if model_type == "fsrcnn":
        print("--------> Results on Test Dataset with Float Model:", PSNR(q_test_results))
    elif model_type == "fcn8":
        print("--------> Results on Test Dataset with Float Model:", q_test_results)


def main():
    args = get_arguments()

    torch.manual_seed(0)
    if (args.device == 'gpu') and (torch.cuda.is_available()) :
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Testing on {device} device.")

    if args.deploy:
        args.test_batch_size = 1

    # ==========================================================================================
    # prepare your data
    # ==========================================================================================
    print("\n[SR7 INFO] Loading Test Data ...")
    
    if args.model_type == "fsrcnn":
        input_dir = os.path.join(args.dataset_dir, 'test/blr')
        label_dir = os.path.join(args.dataset_dir, 'test/gt')
        if args.calib_num_img is None:
            args.calib_num_img = len(os.listdir(input_dir))
        dataset = SuperResolutionDataset(input_dir, label_dir, args.calib_num_img, transform=transforms.ToTensor())
    elif args.model_type == "fcn8":
        input_dir = os.path.join(args.dataset_dir, 'train/image')
        label_dir = os.path.join(args.dataset_dir, 'train/mask')
        dataset = SegmentationDataset(input_dir, label_dir, args.calib_num_img, ratio=1, patch_size=256)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.test_batch_size, shuffle=True, drop_last=True, num_workers=4
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

    # from pytorch_nndct.utils import summary
    input = torch.randn([1, input_shape[0], input_shape[1], input_shape[2]], dtype=torch.float32).to(device)
    print(input.shape)

    # nndct_macs, nndct_params = summary.model_complexity(model, input, return_flops=False, readable=False, print_model_analysis=True)
    quantizer = torch_quantizer(args.quant_mode, model, (input), output_dir=args.quantized_model_dir, device=device)
    model = quantizer.quant_model

    test(args.model_type, model, device, data_loader, criterion)

    if args.quant_mode == 'calib':
        quantizer.export_quant_config()

    if args.deploy:        
        quantizer.export_xmodel(args.quantized_model_dir, deploy_check=True)
        quantizer.export_torch_script(output_dir=args.quantized_model_dir)
        quantizer.export_onnx_model(output_dir=args.quantized_model_dir)

if __name__ == '__main__':
    main()
