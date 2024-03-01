# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class FSRCNN(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(FSRCNN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #FSRCNN::input_0(FSRCNN::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=56, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #FSRCNN::FSRCNN/Sequential[feature_extraction]/Conv2d[0]/ret.3(FSRCNN::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #FSRCNN::FSRCNN/Sequential[feature_extraction]/ReLU[1]/ret.5(FSRCNN::nndct_relu_2)
        self.module_3 = py_nndct.nn.Conv2d(in_channels=56, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #FSRCNN::FSRCNN/Sequential[shrinking]/Conv2d[0]/ret.7(FSRCNN::nndct_conv2d_3)
        self.module_4 = py_nndct.nn.ReLU(inplace=False) #FSRCNN::FSRCNN/Sequential[shrinking]/ReLU[1]/ret.9(FSRCNN::nndct_relu_4)
        self.module_5 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #FSRCNN::FSRCNN/Sequential[mapping]/Conv2d[0]/ret.11(FSRCNN::nndct_conv2d_5)
        self.module_6 = py_nndct.nn.ReLU(inplace=False) #FSRCNN::FSRCNN/Sequential[mapping]/ReLU[1]/ret.13(FSRCNN::nndct_relu_6)
        self.module_7 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #FSRCNN::FSRCNN/Sequential[mapping]/Conv2d[2]/ret.15(FSRCNN::nndct_conv2d_7)
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #FSRCNN::FSRCNN/Sequential[mapping]/ReLU[3]/ret.17(FSRCNN::nndct_relu_8)
        self.module_9 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #FSRCNN::FSRCNN/Sequential[mapping]/Conv2d[4]/ret.19(FSRCNN::nndct_conv2d_9)
        self.module_10 = py_nndct.nn.ReLU(inplace=False) #FSRCNN::FSRCNN/Sequential[mapping]/ReLU[5]/ret.21(FSRCNN::nndct_relu_10)
        self.module_11 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #FSRCNN::FSRCNN/Sequential[mapping]/Conv2d[6]/ret.23(FSRCNN::nndct_conv2d_11)
        self.module_12 = py_nndct.nn.ReLU(inplace=False) #FSRCNN::FSRCNN/Sequential[mapping]/ReLU[7]/ret.25(FSRCNN::nndct_relu_12)
        self.module_13 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #FSRCNN::FSRCNN/Sequential[mapping]/Conv2d[8]/ret.27(FSRCNN::nndct_conv2d_13)
        self.module_14 = py_nndct.nn.ReLU(inplace=False) #FSRCNN::FSRCNN/Sequential[mapping]/ReLU[9]/ret.29(FSRCNN::nndct_relu_14)
        self.module_15 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #FSRCNN::FSRCNN/Sequential[mapping]/Conv2d[10]/ret.31(FSRCNN::nndct_conv2d_15)
        self.module_16 = py_nndct.nn.ReLU(inplace=False) #FSRCNN::FSRCNN/Sequential[mapping]/ReLU[11]/ret.33(FSRCNN::nndct_relu_16)
        self.module_17 = py_nndct.nn.Conv2d(in_channels=16, out_channels=56, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #FSRCNN::FSRCNN/Sequential[expanding]/Conv2d[0]/ret.35(FSRCNN::nndct_conv2d_17)
        self.module_18 = py_nndct.nn.ReLU(inplace=False) #FSRCNN::FSRCNN/Sequential[expanding]/ReLU[1]/ret.37(FSRCNN::nndct_relu_18)
        self.module_19 = py_nndct.nn.ConvTranspose2d(in_channels=56, out_channels=3, kernel_size=[9, 9], stride=[2, 2], padding=[4, 4], output_padding=[1, 1], groups=1, bias=True, dilation=[1, 1]) #FSRCNN::FSRCNN/ConvTranspose2d[deconvolution]/ret(FSRCNN::nndct_conv_transpose_2d_19)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(output_module_0)
        output_module_0 = self.module_11(output_module_0)
        output_module_0 = self.module_12(output_module_0)
        output_module_0 = self.module_13(output_module_0)
        output_module_0 = self.module_14(output_module_0)
        output_module_0 = self.module_15(output_module_0)
        output_module_0 = self.module_16(output_module_0)
        output_module_0 = self.module_17(output_module_0)
        output_module_0 = self.module_18(output_module_0)
        output_module_0 = self.module_19(output_module_0)
        return output_module_0
