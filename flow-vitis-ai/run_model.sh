#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


## clean up previous log files
#rm -f ./log/*.log

# folders
WORK_DIR=./build
LOG_DIR=${WORK_DIR}/0_log

TARGET_102=${WORK_DIR}/../target_zcu102
# ADD YOUR TARGET BOARD HERE

MODEL_DIR=${WORK_DIR}/../input_model
DATASET_DIR=${WORK_DIR}/../dataset

QUANT_DIR=${WORK_DIR}/1_quantize_model
PREDICT_DIR=${WORK_DIR}/2_predictions
COMPILE_DIR=${WORK_DIR}/3_compile_model

PREDICT_FLOAT_DIR=${PREDICT_DIR}/${CNN}/float
PREDICT_QUANT_DIR=${PREDICT_DIR}/${CNN}/quant
PREDICT_GT_DIR=${PREDICT_DIR}/${CNN}/gt
PREDICT_LR_DIR=${PREDICT_DIR}/${CNN}/lr

# logs & results files
PREPARE_DATA_LOG=${CNN}_prepare_data.log
QUANT_LOG=${CNN}_quantize_model.log
EVAL_Q_LOG=${CNN}_evaluate_quantized_model.log
COMP_LOG=${CNN}_compile.log

# files names
# read arguments of the script
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    FLOAT_MODEL_FILENAME=fsrcnn6.h5
    echo "Using default model: ${FLOAT_MODEL_FILENAME}"
else
    FLOAT_MODEL_FILENAME=$1
    echo "Using model: ${FLOAT_MODEL_FILENAME}"
fi

QUANTIZED_MODEL_FILENAME=${FLOAT_MODEL_FILENAME%.*}_quantized.h5
CNN=${FLOAT_MODEL_FILENAME%.*}

##################################################################################



0_clean_and_make_directories() {
    echo " "
    echo "##################################################################################"
    echo "A) CLEAN PREVIOUS DIRECTORIES"
    echo "##################################################################################"
    echo " "
    # clean up previous results
    # rm -rf ${WORK_DIR}; mkdir ${WORK_DIR}
    # rm -rf ${LOG_DIR}; mkdir ${LOG_DIR}
    #rm -rf ${DATASET_DIR}; mkdir ${DATASET_DIR}
    #mkdir  ${DATASET_DIR}/img_calib ${DATASET_DIR}/img_test ${DATASET_DIR}/img_train ${DATASET_DIR}/img_valid
    #mkdir  ${DATASET_DIR}/seg_calib ${DATASET_DIR}/seg_test ${DATASET_DIR}/seg_train ${DATASET_DIR}/seg_valid
    # rm -rf ${QUANT_DIR}; mkdir ${QUANT_DIR}
    # rm -rf ${COMPILE_DIR}; mkdir ${COMPILE_DIR}

    mkdir ${LOG_DIR} ${QUANT_DIR} ${COMPILE_DIR} ${PREDICT_DIR} 2> /dev/null
    mkdir ${LOG_DIR}/${CNN} ${QUANT_DIR}/${CNN} ${COMPILE_DIR}/${CNN} ${PREDICT_DIR}/${CNN} 2> /dev/null
    mkdir ${PREDICT_FLOAT_DIR} ${PREDICT_QUANT_DIR} ${PREDICT_GT_DIR} ${PREDICT_LR_DIR} 2> /dev/null
}


##################################################################################

1_generate_images() {
    echo " "
    echo "##################################################################################"
    echo "Step1: CREATE DATA AND FOLDERS"
    echo "##################################################################################"
    echo " "
    # clean files in pre-built sub-directories
    rm -f ${DATASET_DIR}/test/* ${DATASET_DIR}/train*
    # unzip the original dataset
    unzip ${WORK_DIR}/../dataset.zip -d ${WORK_DIR} >& /dev/null

    # YOU MUST HAVE THE the HDF5 weights file for VGG encoder subnet of FCN8
    cd $MODEL_DIR
    rm -f vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5*  #remove any previous file, if any
    wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 >& /dev/null
    cd ..

}

##################################################################################
2_quantize_model() {
    echo " "
    echo "##########################################################################"
    echo "Step2: CNN QUANTIZATION"
    echo "##########################################################################"
    echo " "
    # display the version of the quantizer
    echo "Using quantizer:"
    pip show -f vai_q_tensorflow2 | grep -E "Name:|Version:"
    echo " "
    #quantize
    cd code
    python vai_q_tensorflow2.py \
        --float_model_file ../${MODEL_DIR}/${FLOAT_MODEL_FILENAME} \
        --quantized_model_file ../${QUANT_DIR}/${CNN}/${QUANTIZED_MODEL_FILENAME} \
        --calib_num_img 1000
    cd ..
}

##################################################################################
# make predictions with quantized model

3_eval_quantized_model() {
    echo " "
    echo "##############################################################################"
    echo "Step3: CNN EVALUATION WITH QUANTIZED MODEL"
    echo "##############################################################################"
    echo " "
    cd code
    echo ../${PREDICT_DIR}/${CNN}
    python eval_quantized_model.py \
	    --float_model_file ../${MODEL_DIR}/${FLOAT_MODEL_FILENAME} \
        --quantized_model_file ../${QUANT_DIR}/${CNN}/${QUANTIZED_MODEL_FILENAME} \
        --eval_num_img 1000 \
        --save_images \
        --save_images_dir ../${PREDICT_DIR}/${CNN}
    cd ..
}


##################################################################################
# Compile xmodel file for ZCU102 board with Vitis AI Compiler
4_compile_vai_zcu102() {
  echo " "
  echo "##########################################################################"
  echo "COMPILE CNN XMODEL FILE WITH Vitis AI for ZCU102"
  echo "##########################################################################"
  echo " "

  vai_c_tensorflow2 \
     --model ${QUANT_DIR}/${CNN}/${QUANTIZED_MODEL_FILENAME} \
     --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json \
	 --output_dir ${COMPILE_DIR}/${CNN} \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}
 }

 5_display_subgraphs() {
   echo " "
   echo "##########################################################################"
   echo "DISPLAY SUBGRAPHS"
   echo "##########################################################################"
   echo " "

   xir png ${COMPILE_DIR}/${CNN}/${FLOAT_MODEL_FILENAME%.*}.xmodel ${COMPILE_DIR}/${CNN}/subgraph_${FLOAT_MODEL_FILENAME%.*}.png

   echo "[INFO] Subgraphs are saved in ${COMPILE_DIR}/${CNN}/subgraph_${FLOAT_MODEL_FILENAME%.*}.png"

  }


##################################################################################
##################################################################################

main() {

    0_clean_and_make_directories

    # create the proper folders and images from the original dataset
    # 1_generate_images #2>&1 | tee ${LOG_DIR}/${CNN}/${PREPARE_DATA_LOG}

    # quantize
    2_quantize_model #2>&1 | tee ${LOG_DIR}/${CNN}/${QUANT_LOG}

    # evaluate post-quantization model
    3_eval_quantized_model #2>&1 | tee ${LOG_DIR}/${CNN}/${EVAL_Q_LOG}

    # compile for ZCU102 board
    4_compile_vai_zcu102 #2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}

    # display subgraphs
    5_display_subgraphs

    # move xmodel file to target board directory
    mkdir ${TARGET_102}/${CNN} 2> /dev/null
    mkdir ${TARGET_102}/${CNN}/model 2> /dev/null
    cp ${COMPILE_DIR}/${CNN}/*.xmodel   ${TARGET_102}/${CNN}/model/
    cp ${COMPILE_DIR}/${CNN}/*.json     ${TARGET_102}/${CNN}/model/

#     copy test images into target board
#     tar -cvf "test.tar" ${DATASET_DIR}/img_test ${DATASET_DIR}/seg_test >& /dev/null
#     gzip -f test.tar
#     cp -f test.tar.gz ${TARGET_190}/
#     cp -f test.tar.gz ${TARGET_280}/
#     cp -f test.tar.gz ${TARGET_102}/
#     cp -f test.tar.gz ${TARGET_104}/

# : '
#     tar -cvf target_zcu102.tar ${TARGET_102}/  >& /dev/null
#     #tar -cvf target_zcu104.tar ${TARGET_104}/  >& /dev/null
#     tar -cvf target_vck190.tar ${TARGET_190}/  >& /dev/null
#     tar -cvf target_vek280.tar ${TARGET_280}/  >& /dev/null
# '
    # cd ../${TARGET_102}/${CNN}/model/
    output_folder=${COMPILE_DIR}/${CNN}/
    # cd ../../../

    echo " "
    echo "#####################################"
    echo "MAIN CNN FLOW COMPLETED"
    echo "#####################################"
    echo "Find the output files in the folders:"
    echo "---> ${output_folder}"
    echo "#####################################"



}

main
