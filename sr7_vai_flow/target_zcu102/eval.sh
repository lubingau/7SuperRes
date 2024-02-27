#!/bin/bash

# Evaluate the performance of the CNN application by measuring PSNR 

mkdir build 2> /dev/null
bash -x ./build_eval.sh
mv code/build_eval build/eval
./build/eval "../sr7_dataset/test/blr" "../sr7_dataset/test/gt" "/home/petalinux/target_zcu102/fsrcnn6_relu/model/fsrcnn6_relu.xmodel"
