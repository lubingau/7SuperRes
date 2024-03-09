#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

## Author: Daniele Bagni, AMD/Xilinx Inc

## date 10 Aug 2023

# REMEMBER THAT $1 is the "main" routine
# LOG_FILENAME=$2
# MODEL_NAME=$3


# echo " "
# echo "==========================================================================="
# echo "WARNING: "
# echo "  'run_all.sh' MUST ALWAYS BE LAUNCHED BELOW THE 'files' FOLDER LEVEL "
# echo "  (SAME LEVEL OF 'scripts' AND 'target' FOLDER)                       "
# echo "  AS IT APPLIES RELATIVE PATH AND NOT ABSOLUTE PATHS                  "
# echo "==========================================================================="
# echo " "

# read arguments of the script
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    FLOAT_MODEL_FILENAME="fsrcnn6_relu_35ep.pt"
    MODEL_TYPE="fsrcnn"
    echo "Using default model: ${FLOAT_MODEL_FILENAME}"
else
    MODEL_TYPE=$1
    FLOAT_MODEL_FILENAME=$2
    echo "Using model: ${FLOAT_MODEL_FILENAME}"
fi

CNN=${FLOAT_MODEL_FILENAME%.*}
QUANTIZED_MODEL_FILENAME="FSRCNN_int.pt"
ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json"
	
# folders
WORK_DIR=./build

TARGET_102=${WORK_DIR}/../target_zcu102
# ADD YOUR TARGET BOARD HERE

MODEL_DIR=${WORK_DIR}/../input_model
DATASET_DIR=../supres_dataset

LOG_DIR=${WORK_DIR}/0_log
QUANT_DIR=${WORK_DIR}/1_quantize_model
PREDICT_DIR=${WORK_DIR}/2_predictions
COMPILE_DIR=${WORK_DIR}/3_compile_model

PREDICT_FLOAT_DIR=${PREDICT_DIR}/${CNN}/float
PREDICT_QUANT_DIR=${PREDICT_DIR}/${CNN}/quant
PREDICT_LABEL_DIR=${PREDICT_DIR}/${CNN}/label
PREDICT_INPUT_DIR=${PREDICT_DIR}/${CNN}/input

# logs & results files
QUANT_LOG=${CNN}_quantize_model.log
EVAL_Q_LOG=${CNN}_evaluate_quantized_model.log
COMP_LOG=${CNN}_compile.log

##################################################################################

0_clean_and_make_directories() {
    echo " "
    echo "##################################################################################"
    echo "A) CLEAN PREVIOUS DIRECTORIES"
    echo "##################################################################################"
    echo " "

    # clean up previous log files
    rm -f ${LOG_DIR}/${CNN}/*.log

    mkdir ${LOG_DIR} ${QUANT_DIR} ${COMPILE_DIR} ${PREDICT_DIR} 2> /dev/null
    mkdir ${LOG_DIR}/${CNN} ${QUANT_DIR}/${CNN} ${COMPILE_DIR}/${CNN} ${PREDICT_DIR}/${CNN} 2> /dev/null
    mkdir ${PREDICT_FLOAT_DIR} ${PREDICT_QUANT_DIR} ${PREDICT_LABEL_DIR} ${PREDICT_INPUT_DIR} 2> /dev/null
}


# ===========================================================================
# STEP4: Vitis AI Quantization of ResNet18 on VCoR
# ===========================================================================
1_quantize_model(){
    echo " "
    echo "----------------------------------------------------------------------------------"
    echo "[SR7 INFO] QUANTIZE VCoR TRAINED CNN"
    echo "----------------------------------------------------------------------------------"
    # bash -x ./scripts/run_quant.sh

    #WEIGHTS=./pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0/float
    # WEIGHTS=./build/float
    # GPU_ID=0
    #QUANT_DIR=${QUANT_DIR:-quantized}
    # QUANT_DIR=./build/quantized
    export PYTHONPATH=${PWD}:${PYTHONPATH}

    cd code
    echo "Conducting Quantization"

    # fix calib
    echo "-----------------------------------------------------------"
    echo "------------------------CALIBRATION------------------------"
    echo "-----------------------------------------------------------"
    python vai_q_pytorch.py \
        --model_type ${MODEL_TYPE} \
        --float_model_file ../${MODEL_DIR}/${FLOAT_MODEL_FILENAME} \
        --quantized_model_dir ../${QUANT_DIR}/${CNN} \
        --dataset_dir ../${DATASET_DIR} \
        --quant_mode calib \
        --calib_num_img 500

    # fix test
    echo "----------------------------------------------------"
    echo "------------------------TEST------------------------"
    echo "----------------------------------------------------"
    python vai_q_pytorch.py \
        --model_type ${MODEL_TYPE} \
        --float_model_file ../${MODEL_DIR}/${FLOAT_MODEL_FILENAME} \
        --quantized_model_dir ../${QUANT_DIR}/${CNN} \
        --dataset_dir ../${DATASET_DIR} \
        --quant_mode test \
        --calib_num_img 500
    
    # deploy
    echo "------------------------------------------------------"
    echo "------------------------DEPLOY------------------------"
    echo "------------------------------------------------------"
    python vai_q_pytorch.py \
        --model_type ${MODEL_TYPE} \
        --float_model_file ../${MODEL_DIR}/${FLOAT_MODEL_FILENAME} \
        --quantized_model_dir ../${QUANT_DIR}/${CNN} \
        --dataset_dir ../${DATASET_DIR} \
        --quant_mode test \
        --calib_num_img 500 \
        --deploy
}

##################################################################################
# make predictions with quantized model
2_eval_quantized_model() {
    echo " "
    echo "##############################################################################"
    echo "Step3: CNN EVALUATION WITH QUANTIZED MODEL"
    echo "##############################################################################"
    echo " "
    cd code
    echo ../${PREDICT_DIR}/${CNN}
    python eval_quantized_model.py \
        --model_type ${MODEL_TYPE} \
	    --float_model_file ../${MODEL_DIR}/${FLOAT_MODEL_FILENAME} \
        --quantized_model_file ../${QUANT_DIR}/${CNN}/${QUANTIZED_MODEL_FILENAME} \
        --dataset_dir ../${DATASET_DIR} \
        --eval_num_img 3000 \
        --save_images \
        --save_images_dir ../${PREDICT_DIR}/${CNN}
    cd ..
}

# ===========================================================================
# STEP5: Vitis AI Compile ResNet18 VCoR for Target Board
# ===========================================================================
3_compile_vai_zcu102(){
    echo " "
    echo "----------------------------------------------------------------------------------"
    echo "[DB INFO STEP5] COMPILE VCoR QUANTIZED CNN"
    echo "----------------------------------------------------------------------------------"
    echo " "
    QUANTIZED_XMODEL_MODEL_FILENAME=${QUANTIZED_MODEL_FILENAME%.*}.xmodel
    vai_c_xir \
        --xmodel          ${QUANT_DIR}/${CNN}/${QUANTIZED_XMODEL_MODEL_FILENAME} \
        --arch            ${ARCH} \
        --output_dir      ${COMPILE_DIR}/${CNN} \
        --net_name        ${CNN}
    #	--options         "{'mode':'debug'}"
    #  --options         '{"input_shape": "1,224,224,3"}'
}

##################################################################################
# Display subgraphs
4_display_subgraphs() {
   echo " "
   echo "##########################################################################"
   echo "DISPLAY SUBGRAPHS"
   echo "##########################################################################"
   echo " "

   xir png ${COMPILE_DIR}/${CNN}/${FLOAT_MODEL_FILENAME%.*}.xmodel ${COMPILE_DIR}/${CNN}/subgraph_${FLOAT_MODEL_FILENAME%.*}.png

   echo "[INFO] Subgraphs are saved in ${COMPILE_DIR}/${CNN}/subgraph_${FLOAT_MODEL_FILENAME%.*}.png"

  }


# ===========================================================================
# main
# ===========================================================================

# do not change the order of the following commands

pip install randaugment
pip install torchsummary

0_clean_and_make_directories
1_quantize_model
2_eval_quantized_model
3_compile_vai_zcu102
4_display_subgraphs