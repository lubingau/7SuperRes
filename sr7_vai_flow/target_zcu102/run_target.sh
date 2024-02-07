#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# clean everything
clean() {
    rm -rf ./build ./rpt ./png*
    rm -f ./run_cnn get_dpu_fps *.png
    mkdir ./png_fcn8
    mkdir ./png_fcn8ups
    mkdir ./png_unet
    mkdir ./rpt
    ##remove images
    rm -rf dataset
}

# build  test images
dataset() {
    echo " "
    echo "##################################################################################"
    echo "EXTRACT DATASET"
    echo "##################################################################################"
    echo " "

    unzip ../sr7_dataset.zip -d ./

    echo "[DZ INFO] Dataset extracted"
}

# compile CNN application
compile() {
    echo " "
    echo "##################################################################################"
    echo "COMPILE CNN APPLICATION"
    echo "##################################################################################"
    echo " "
    cd code
    bash -x ./build_app.sh
    mv code ../run_cnn
    # bash -x ./build_get_dpu_fps.sh
    # mv code ../get_dpu_fps
    cd ..
    echo "[DZ INFO] CNN application compiled"
}

# now run semantic segmentation with 3 CNNs using VART C++ APIs with single thread
run_models() {
    ./run_cnn ./fsrcnn6_relu/model/fsrcnn6_relu.xmodel  ../sr7_dataset/test/blr/ 6 0 1 2> /dev/null | tee ./rpt/logfile_cpp_fsrcnn6_relu.txt
}

run_fps() {
    # get the fps performance  with multithreads
    bash -x ./code/run_cnn_fps.sh 2> /dev/null | tee ./rpt/logfile_fps.txt
}


#clean
if [ ! -d "dataset" ]; then
dataset
fi
if [ "$1" != "--no-compile" ]; then
compile
fi
mkdir outputs 2> /dev/null
run_models
#run_fps
