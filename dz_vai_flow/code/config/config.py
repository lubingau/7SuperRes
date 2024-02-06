#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
'''

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


import os

###############################################################################
# project folders
###############################################################################

def get_script_directory():
    path = os.getcwd()
    return path

# get current directory
SCRIPT_DIR = get_script_directory()
print("config.py runs from ", SCRIPT_DIR)

# dataset top level folder
DATASET_DIR = os.path.join(SCRIPT_DIR, "../../dz_dataset")

# output directories
dir_train_input = os.path.join(DATASET_DIR, "train/blr")
dir_train_label = os.path.join(DATASET_DIR, "train/gt")
dir_test_input  = os.path.join(DATASET_DIR, "test/blr")
dir_test_label  = os.path.join(DATASET_DIR, "test/gt")