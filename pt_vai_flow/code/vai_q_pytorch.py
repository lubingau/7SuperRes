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
from sr_model import FSRCNN
from data_loader import SuperResolutionDataset

# ==========================================================================================
# Get Input Arguments
# ==========================================================================================
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Vitis AI Pytorch Quantization")

    # model config
    parser.add_argument("--float_model_file", type=str,
                        help="pt floating point file path name")
    # quantization config
    parser.add_argument("--quantized_model_dir", type=str,
                        help="quantized model file path name ")
    # dataset directory
    parser.add_argument("--dataset_dir", type=str, default="../../supres_dataset",
                        help="dataset directory")
    # test batch size
    parser.add_argument("--test_batch_size", type=int, default=128,
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

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    n_batches = 0
    n_batches_max = 10000
    with torch.no_grad():
        for blr_data, gt_data in test_loader:
            if n_batches >= n_batches_max:
                break
            blr_images, gt_images = blr_data.to(device), gt_data.to(device)
            sr_images = model(blr_images)
            
            test_loss += F.mse_loss(sr_images, gt_images).item()
            n_batches += 1
    
    test_loss /= len(test_loader.dataset)
    psnr = 10 * np.log10(1 / test_loss)
    print('\nSuper-resolution Test set: Average loss: {:.4f}'.format(test_loss))
    print('Super-resolution Test set: Average PSNR: {:.4f}\n'.format(psnr))


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

    transform = transforms.Compose([transforms.ToTensor()])

    blr_dir = os.path.join(args.dataset_dir, 'test/blr')
    gt_dir = os.path.join(args.dataset_dir, 'test/gt')
    if args.calib_num_img is None:
        args.calib_num_img = len(os.listdir(blr_dir))
    
    super_resolution_dataset = SuperResolutionDataset(blr_dir, gt_dir, args.calib_num_img, transform)

    super_resolution_loader = torch.utils.data.DataLoader(
        super_resolution_dataset, batch_size=args.test_batch_size, shuffle=True, drop_last=True, num_workers=4
    )

    # Get the input shape from the first data of the super_resolution_loader
    for blr_data, gt_data in super_resolution_loader:
        input_shape = blr_data.shape[1:]
        break
    print(input_shape)
    upscale_factor = 2
    color_channels = input_shape[0]

    model = FSRCNN(d=56, s=16, m=6, input_size=input_shape,   upscaling_factor=upscale_factor, color_channels=color_channels)

    model.load_state_dict(torch.load(args.float_model_file , map_location=device), strict=True)

    summary(model, input_shape)

    # from pytorch_nndct.utils import summary
    input = torch.randn([1, input_shape[0], input_shape[1], input_shape[2]], dtype=torch.float32).to(device)
    print(input.shape)

    # nndct_macs, nndct_params = summary.model_complexity(model, input, return_flops=False, readable=False, print_model_analysis=True)
    quantizer = torch_quantizer(args.quant_mode, model, (input), output_dir=args.quantized_model_dir, device=device)
    model = quantizer.quant_model

    test(model, device, super_resolution_loader)

    if args.quant_mode == 'calib':
        quantizer.export_quant_config()

    if args.deploy:        
        quantizer.export_xmodel(args.quantized_model_dir, deploy_check=True)
        quantizer.export_torch_script(output_dir=args.quantized_model_dir)
        quantizer.export_onnx_model(output_dir=args.quantized_model_dir)

if __name__ == '__main__':
    main()
