#!/bin/bash

# Evaluate the performance of the CNN application by measuring PSNR 

if [ "$1" = "--no-build" ]; then
    echo "[SR7 INFO] Skipping evaluation application compilation"
elif [ $# -eq 0 ]; then
    echo "[SR7 INFO] Compiling evaluation application"
    mkdir build 2> /dev/null
    bash -x ./code/build_eval.sh
    echo " "
    mv code/code build/eval
else
    echo "Usage: ./eval.sh [--no-build]"
    exit 1
fi
echo " "
./build/eval "../sr7_dataset/test/blr" "../sr7_dataset/test/gt" "/home/petalinux/target_zcu102/fsrcnn6_relu/model/fsrcnn6_relu.xmodel"

