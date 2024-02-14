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
    echo " EXTRACT DATASET"
    echo "##################################################################################"
    echo " "


    if [ ! -d "../sr7_dataset" ]; then
        if [ ! -f "../sr7_dataset.zip" ]; then
            echo "[SR7 INFO] Dataset not found"
            echo "[SR7 INFO] Please import the dataset and place it in the /home/petalinux/ folder"
            exit 1
        else
            unzip ../sr7_dataset.zip -d ../
            echo "[SR7 INFO] Dataset extracted"
        fi
    else
        echo "[SR7 INFO] Dataset already exists"
    fi
}

# compile CNN application
compile() {
    echo " "
    echo "##################################################################################"
    echo " COMPILE CNN APPLICATION"
    echo "##################################################################################"
    echo " "

    if [ "$1" = true ]; then
        echo "[SR7 INFO] Compiling CNN application"
        cd code
        #bash -x ./build_app.sh
        g++ -o build/SuperRes7 src/SuperRes7.cpp `pkg-config --cflags --libs opencv4`
        g++ -o build/build_mask src/build_mask.cpp `pkg-config --cflags --libs opencv4`
        #mv code build/SuperRes7
        # bash -x ./build_get_dpu_fps.sh
        # mv code ../get_dpu_fps
        cd ..
        echo "[SR7 INFO] CNN application compiled"
    else
        echo "[SR7 INFO] Skipping CNN application compilation"
    fi
}

build_mask() {
    echo " "
    echo "##################################################################################"
    echo " BUILD MASK"
    echo "##################################################################################"
    echo " "

    png_file = $1
    echo $png_file
    if [ ! -d "mask" ]; then
        mkdir mask
    fi
    cd mask/
    ./../code/build/build_mask $png_file
    cd ..
}

# now run the CNN model
run_models() {
    echo " "
    echo "##################################################################################"
    echo " RUN CNN MODELS"
    echo "##################################################################################"
    echo " "

    rm -r outputs 2> /dev/null
    rm -r inputs 2> /dev/null
    mkdir outputs
    mkdir inputs

    echo "[SR7 INFO] Running CNN model"
    #./run_cnn ./fsrcnn6_relu/model/fsrcnn6_relu.xmodel  ../sr7_dataset/test/blr/ 1 0 1 200 #2> /dev/null | tee ./rpt/logfile_cpp_fsrcnn6_relu.txt
    ./code/build/SuperRes7 $1 $2 $3 $4
}

run_fps() {
    # get the fps performance  with multithreads
    bash -x ./code/run_cnn_fps.sh 2> /dev/null | tee ./rpt/logfile_fps.txt
}

# MAIN
if [ "$1" = "--build" ]; then
    compilation=true
    cd code/
    rm -rf build/
    mkdir build
    cd ..
    echo "[SR7 INFO] Deleted build folder, ready for compilation"
elif [ "$1" = "--no-build" ]; then
    compilation=false
else
    echo "Usage: [--build | --no-build] [png_file] [patch_size] [stride] [mask_file]"
    exit 1
fi

png_file=$2
patch_size=$3
stride=$4
mask_file=$5

#clean
#dataset
compile $compilation
build_mask $png_file
run_models $png_file $patch_size $stride $mask_file
#run_fps