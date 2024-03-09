#!/bin/bash

# Evaluate the performance of the CNN application by measuring PSNR 

DEBUG_DIR=./debug
DEBUG_EVAL=${DEBUG_DIR}/eval

# compile CNN application
compile() {
    echo " "
    echo "##################################################################################"
    echo " COMPILE EVALUATION APPLICATION"
    echo "##################################################################################"
    echo " "

    if [ "$1" = true ]; then
        echo "[SR7 INFO] Compiling evaluation application"
        cd code
        bash -x ./build_eval.sh
        # g++ -o code src/eval.cpp `pkg-config --cflags --libs opencv4` for testing on local machine without Vitis AI
        mv code ../build/eval
        cd ..
        echo "[SR7 INFO] Evaluation application compiled"
    else
        echo "[SR7 INFO] Skipping evaluation application compilation"
    fi
}

# now run the CNN models
run_models() {
    echo " "
    echo "##################################################################################"
    echo " RUN EVALUATION MODELS"
    echo "##################################################################################"
    echo " "

    mkdir outputs

    echo "[SR7 INFO] Running CNN models"
    ./build/eval $1 $2 $3 $4 $5 $6 $7 $8 $9

    echo " "
    echo "[SR7 INFO] Application finished"
}

# PARSER

# Display help message
display_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -n,  --no-compilation   Disable compilation (default: true)"
    echo "  -i,  --input-dir        Specify the input patches directory (default: ../supres_dataset/test/)"
    echo "  -xr, --xmodel-supres    Specify the path of the super-resolution model (default: models/fsrcnn6_relu/fsrcnn6_relu.xmodel)"
    echo "  -xs, --xmodel-segment   Specify the path of the segmentation model (default: models/fcn8/fcn8.xmodel)"
    echo "  -t,  --num-threads      Specify the number of threads (default: 6)"
    echo "  -r,  --num-processes    Specify the number of processes (default: 4)"
    echo "  -h,  --help             Display this help message"
}

# Default values
compilation=true
input_dir="../supres_dataset/test"
path_xmodel_supres="models/fsrcnn6_relu/fsrcnn6_relu.xmodel"
path_xmodel_segment="models/fcn8/fcn8.xmodel"
num_threads=6
num_processes=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -n|--no-compilation)
            compilation=false
            shift
            ;;
        -i|--input-dir)
            input_dir="$2"
            shift
            shift
            ;;
        -xr|--xmodel-supres)
            path_xmodel_supres="$2"
            shift
            shift
            ;;
        -xs|--xmodel-segment)
            path_xmodel_segment="$2"
            shift
            shift
            ;;
        -t|--num-threads)
            num_threads="$2"
            shift
            shift
            ;;
        -r|--num-processes)
            num_processes="$2"
            shift
            shift
            ;;
        -h|--help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

compile $compilation
run_models $path_xmodel_supres $path_xmodel_segment $num_threads $num_processes

