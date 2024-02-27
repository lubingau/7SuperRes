#!/bin/bash

# Evaluate the performance of the CNN application by measuring PSNR 

mkdir build 2> /dev/null
g++ -o build/eval code/src/eval.cpp `pkg-config --cflags --libs opencv4` 
./build/eval "../sr7_dataset/test/blr" "../sr7_dataset/test/gt"
