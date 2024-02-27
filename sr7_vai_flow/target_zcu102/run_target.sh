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
        mkdir build 2> /dev/null
        # bash -x ./build_app.sh
        g++ -o build/SuperRes7 src/SuperRes7.cpp `pkg-config --cflags --libs opencv4`
        mv code build/SuperRes7
        # g++ -o build/build_mask src/build_mask.cpp `pkg-config --cflags --libs opencv4`
        # bash -x ./build_get_dpu_fps.sh
        # mv code ../get_dpu_fps
        cd ..
        echo "[SR7 INFO] CNN application compiled"
    else
        echo "[SR7 INFO] Skipping CNN application compilation"
    fi
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

    echo "[SR7 INFO] Running memory footprint"
    ./code/src/memory_footprint.sh &
    pid_memory_footprint=$!
    echo "[SR7 INFO] Running CNN model"
    ./code/build/SuperRes7 $1 $2 $3 $4 $5 "debug/"

    kill "$pid_memory_footprint"
    echo "[SR7 INFO] Saved memory footprint in code/src/memory_usage.csv"
}

run_fps() {
    # get the fps performance  with multithreads
    bash -x ./code/run_cnn_fps.sh 2> /dev/null | tee ./rpt/logfile_fps.txt
}

# MAIN

if [ "$#" = 0 ]; then
    echo "No arguments given. Using default parameters."
    echo "Write --help or -h to get more informations."
    compilation=true
    png_file="../blr_sensor_image_crop_0.png"
    patch_size=128
    stride=0.9
    output_dir="outputs/"
    path_xmodel="/home/eau_kipik/SuperRes7/sr7_vai_flow/fsrcnn6_relu.xmodel" #"/home/petalinux/target_zcu102/fsrcnn6_relu/model/fsrcnn6_relu.xmodel"

elif [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ "$1" = "--h" ]; then
    echo "Usage: [--build | --no-build] [png_file] [patch_size] [stride] [output_dir] [--debug]"
    exit 1 

else
    if [ "$1" = "--build" ]; then
        compilation=true
        rm -rf code/build/
        mkdir code/build
        echo "[SR7 INFO] Deleted build folder, ready for compilation"
    elif [ "$1" = "--no-build" ]; then
        compilation=false
    else
        echo "Usage: [--build | --no-build] [png_file] [patch_size] [stride] [output_dir] [--debug]"
        exit 1
    fi
    if [ "$6" = "--debug" ]; then
        echo "[SR7 INFO] Debug mode"
        rm -rf debug
        mkdir debug
        cd debug
        mkdir patcher runCNN runCNN/input runCNN/output rebuilder eval eval/input eval/output
        cd ..
    fi

    png_file=$2
    patch_size=$3
    stride=$4
    output_dir=$5
    path_xmodel="/home/petalinux/target_zcu102/fsrcnn6_relu/model/fsrcnn6_relu.xmodel"    
fi

#clean
#dataset
compile $compilation
run_models $png_file $patch_size $stride $path_xmodel $output_dir
#run_fps