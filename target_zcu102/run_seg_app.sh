#!/bin/bash


# logs & results files
OUTPUT_DIR=./outputs_seg
BUILD_DIR=./build_seg
LOG_DIR=./log_seg
DEBUG_DIR=./debug_seg

COMPILE_LOG=${LOG_DIR}/compile.log
RUN_LOG=${LOG_DIR}/run.log

DEBUG_PATCHER=${DEBUG_DIR}/patcher
DEBUG_CNN=${DEBUG_DIR}/cnn
DEBUG_REBUILDER=${DEBUG_DIR}/rebuilder

# clean everything
clean() {
    echo " "
    echo "##################################################################################"
    echo " CLEAN APPLICATION"
    echo "##################################################################################"
    echo " "
    if [ "$1" = true ]; then
        rm -rf ${OUTPUT_DIR}
        rm -rf ${BUILD_DIR}
        rm -rf ${LOG_DIR}
        rm -rf ${DEBUG_DIR}
        mkdir ${OUTPUT_DIR}
        mkdir ${BUILD_DIR}
        mkdir ${LOG_DIR}
        mkdir ${DEBUG_DIR} ${DEBUG_PATCHER} ${DEBUG_CNN} ${DEBUG_REBUILDER}
        echo "[SR7 INFO] Application cleaned"
    else
        echo "[SR7 INFO] Skipping clean application"
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
        bash -x ./build_seg_app.sh
        # g++ -o code src/SuperSeg7.cpp `pkg-config --cflags --libs opencv4` for testing on local machine without Vitis AI
        mv code ../${BUILD_DIR}/SuperSeg7
        cd ..
        echo "[SR7 INFO] CNN application compiled"
    else
        echo "[SR7 INFO] Skipping CNN application compilation"
    fi
}

# now run the CNN models
run_models() {
    echo " "
    echo "##################################################################################"
    echo " RUN CNN MODELS"
    echo "##################################################################################"
    echo " "

    echo "[SR7 INFO] Running memory footprint"
    rm -f ${OUTPUT_DIR}/memory_usage.csv
    rm -f ${OUTPUT_DIR}/execution_times.txt
    ./code/src/memory_footprint.sh &  # launch memory footprint in background
    pid_memory_footprint=$!

    echo "[SR7 INFO] Running CNN models"
    ./${BUILD_DIR}/SuperSeg7 $1 $2 $3 $4 $5 $6 $7 $8

    kill "$pid_memory_footprint"
    mv memory_usage.csv ${OUTPUT_DIR}/memory_usage.csv
    mv execution_times.txt ${OUTPUT_DIR}/execution_times.txt
    echo "[SR7 INFO] Saved memory footprint in ${OUTPUT_DIR}/memory_usage.csv"
    echo "[SR7 INFO] Saved execution times in ${OUTPUT_DIR}/execution_times.txt"
    echo " "
    echo "[SR7 INFO] Application finished"
}

# PARSER

# Display help message
display_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -n,  --no-compilation   Disable compilation (default: true)"
    echo "  -i,  --input            Specify the input image (default: input/sensor_image.png)"
    echo "  -p,  --patch-size       Specify the patch size (default: 128)"
    echo "  -s,  --stride           Specify the stride (default: 0.9)"
    echo "  -o,  --output-dir       Specify the output directory (default: outputs_seg/)"
    echo "  -xr, --xmodel-supres    Specify the path of the super-resolution model (default: models/fsrcnn6_relu/fsrcnn6_relu.xmodel)"
    echo "  -xs, --xmodel-segment   Specify the path of the segmentation model (default: models/fcn8/fcn8.xmodel)"
    echo "  -t,  --num-threads      Specify the number of threads (default: 6)"
    echo "  -r,  --num-processes    Specify the number of processes (default: 4)"
    echo "  -h,  --help             Display this help message"
}


# Default values
compilation=true
input_img="input/sensor_image.png"
patch_size=256
stride=0.9
output_dir=${OUTPUT_DIR}
path_xmodel_supres="models/fsrcnn6_relu/fsrcnn6_relu.xmodel"
path_xmodel_segment="models/fcn8/fcn8.xmodel"
num_threads=6
num_processes=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--no-compilation)
            compilation=false
            ;;
        -i|--input)
            shift
            input_img="$1"
            ;;
        -p|--patch-size)
            shift
            patch_size="$1"
            ;;
        -s|--stride)
            shift
            stride="$1"
            ;;
        -o|--output-dir)
            shift
            output_dir="$1"
            ;;
        -xr|--xmodel-supres)
            shift
            path_xmodel_supres="$1"
            ;;
        -xs|--xmodel-segment)
            shift
            path_xmodel_segment="$1"
            ;;
        -t|--num-threads)
            shift
            num_threads="$1"
            ;;
        -r|--num-processes)
            shift
            num_processes="$1"
            ;;
        -h|--help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Usage of argument values
echo "\n[SR7 INFO] Running with the following parameters:"
echo "   > Compilation: $compilation"
echo "   > Input image: $input_img"
echo "   > Patch size: $patch_size"
echo "   > Stride: $stride"
echo "   > Output directory: $output_dir"
echo "   > Path of the super-resolution model: $path_xmodel_supres"
echo "   > Path of the segmentation model: $path_xmodel_segment"
echo "   > Number of threads: $num_threads"
echo "   > Number of processes: $num_processes"


clean $compilation
compile $compilation 2>&1 | tee ${COMPILE_LOG}
run_models $input_img $patch_size $stride $path_xmodel_segment $output_dir $num_threads $num_processes ${DEBUG_DIR} 2>&1 | tee ${RUN_LOG}
