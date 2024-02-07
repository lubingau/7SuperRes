![ZCU102](./doc/logo_SuperRes7.png)

# A super resolution AI on ZCU102

The SuperRes7 project is part of an Industrial Study Project (BEI) of ENSEEIHT. The objective of this project is to develop an embedded application for satellite image super-resolution using AI techniques. Ultimately, the application should be capable of being deployed on an already-orbiting satellite. The system is a sandbox (thus reconfigurable) and is equipped with an optical sensor capable of capturing RGB images with a resolution of 150 million pixels (14192x10640).


# How to train the model

Download our dataset [here](https://drive.google.com/drive/folders/1xJYEhfPTt9Ox6RwFbfRrWGRmBvW3IX2l?usp=sharing) (extracted from [Cars Overhead With Context](https://gdo152.llnl.gov/cowc/)) and copy it in the SuperRes7 repo.
You should have this structure:
```
SuperRes7
├── sr7_dataset
│   ├── test
│   │   ├── blr
│   │   └── gt
│   └── train
│       ├── blr
│       └── gt
├── sr7_vai_flow
│   ├── build
│   ├── code
│   ├── input_model
│   └── target_zcu102
├── doc
├── pkgs
└── training
```

To begin with, you need to install the required packages:
```bash
pip3 install -r requirements.txt
```

Our model is an FSRCNN adapted from "*[Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367)*" with the following parameters:

| D  | S  | M |
|----|----|---|
| 56 | 16 | 6 |

<br>
To train our model, please run this Python script:

```bash
python3 FSRCNN_train.py
```

Here are all options :
```bash
options:
  --dataset_path DATASET_PATH   Path to the dataset
  --params_path PARAMS_PATH     Path to the training parameters in the .json file
```

## How to train with your own dataset

We provide a Python script that allows you to create your dataset with your own images:
```bash
python3 create_dataset.py --input_dir path-to-your-images
```

Here are all options :

```bash
options:
  --input_dir INPUT_DIR       Path to input image directory
  --hr_dir HR_DIR             Path to save high-resolution patches
  --blr_dir BLR_DIR           Path to save blurred low res patches
  --kernel_path KERNEL_PATH   Path to blur kernel
  --stride STRIDE             Stride ratio for patch extraction
  --patch_size PATCH_SIZE     Size of the patches to extract
  --nb_img NB_IMG             Number of images to process, -1 for full dataset
```
<br>

# How to compile the model with Vitis-AI

Once the model is trained, it needs to be quantized and compiled with Vitis AI tools. We use version 3.0. Here is a schema representing our Vitis AI flow:

![Flow Vitis AI](./doc/sr7_vai_flow.png)

## Requirements
- Ubuntu 22.04 host PC
- Vitis AI 3.0 repository
- Docker

## Vitis AI installation
1. Clone or download the [Vitis AI 3.0](https://github.com/Xilinx/Vitis-AI/tree/3.0) repository. ***WARNING** : be careful to use the 3.0 version*.
    ```bash
    git clone https://github.com/Xilinx/Vitis-AI
    git checkout -b 3.0 origin/3.0
    ```
2. Build the Vitis-AI docker on your PC
    ```bash
    cd Vitis-AI/
    cd docker/
    sudo ./docker_build.sh -t cpu -f tf2
    ```
    This operation might take some time (~20 minutes). If you want to use gpu instead of cpu, you can but we haven't tried it
3. Once the process is finished, you should see something like this with the command ```sudo docker images```:
    ```text
    REPOSITORY                        TAG      IMAGE ID       CREATED         SIZE
    xilinx/vitis-ai-tensorflow2-cpu   latest   e1501ac96fd0   10 days ago     6.75GB
    ```
4. You will need to install some missing packages and libraries into the Vitis AI container. Copy the file `setup_docker_env.sh` into the `Vitis-AI` folder
    ```bash
    cd SuperRes7
    cp setup_docker_env.sh ../Vitis-AI/
    cp -r pkgs/ ../Vitis-AI/src/vai_quantizer/vai_q_tensorflow2.x/
    ```
5. Copy the `sr7_vai_flow` and `sr7_dataset` folders into the `Vitis-AI` folder:
    ```bash
    cp -r sr7_vai_flow/ sr7_dataset/ ../Vitis-AI/
    ```
6. To launch the docker container with Vitis AI tools, execute the following commands from the `Vitis-AI` folder:
    ```bash
    cd Vitis-AI/
    sudo ./docker_run.sh xilinx/vitis-ai-tensorflow2-cpu:latest
    ```
7. Once you are into the Vitis AI container, execute the following script to install some missing packages and libraries:
    ```bash
    sudo ./setup_docker_env.sh
    ```
8. Voilà! You are ready to play with Vitis AI tools.

***WARNING**: you will need to execute the `setup_docker_env.sh` script each time you launch the Vitis AI container.*

<br>


# Run the Vitis-AI flow
1. Launch the Vitis AI container. Be careful to execute `setup_docker_env.sh` script.
2. Launch the flow :
    ```bash
    cd sr7_vai_flow/
    ./run_model.sh
    ```
3. Output files will be in `build/`:
    - `0_log`: all log
    - `1_quantize_model`: quantize model
    - `2_predictions`: all predictions (png files)
    - `3_compile_model`: output compiled model and subgraphs
4. Output files will be automatically copy into `target_zcu102` folder

---
*You can verify that the compilation is successful by viewing the `subgraph_<model>.png` file. This file represents the content of the `.xmodel` file which contains all the information required by the DPU. The file will be executable if and only if there is **only one DPU block** (blue). If there are multiple DPU blocks, it means that some operations are performed on the CPU within the model. This could be due to a mathematical operation not supported by the DPU, such as an activation function for example. Check the Vitis AI logs as well as the [Vitis AI documentation](https://docs.xilinx.com/r/3.0-English/ug1414-vitis-ai/Currently-Supported-Operators).*

## Compiled your model

You can quantize and compile your own model. To do this, please copy your trained model in TensorFlow 2 (the latest version of TensorFlow) into the `input_model` folder. Then, execute the Vitis AI flow by specifying the name of your model. The flow will take care of creating a folder for this model for all key steps of the flow.

```bash
./run_model.sh your_model.h5
```
You may have to make the file executable with `chmod +x run_model.sh`

## Run on the ZCU102

1. Flash [ZCU102 DPU image](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu102-dpu-v2022.2-v3.0.0.img.gz) (or download it [here](https://drive.google.com/file/d/17IEiRW8wZ8UVISNuKscxq4__2BXgZSsM/view?usp=sharing)) on the SD card with balenaEtcher for example. Then, insert the SD card into the ZCU102, and switch on the board. Be careful to put correctly boot mode switches on the ZCU102: 1000.

2. When you power on the ZCU102, a red led appear, and then disappear when boot is done. You can connect to the board with UART or ethernet. Prefer the ethernet connection for file transfer, otherwise you will have to remove the SD card.

3. Copy the `target_zcu102` folder from the host PC to the ZCU102:
    ```bash
    cd Vitis-AI/sr7_vai_flow/
    scp -r target_zcu102/ petalinux@192.168.1.64:/home/petalinux/
    ```
    Replace the IP address with yours.
4. Connect as a root (necessary for compilation):
    ``` bash
    sudo -s
    ```
5. Copy `sr7_dataset.zip` on the ZCU102:
    ``` bash
    scp sr7_dataset.zip petalinux@192.168.1.64:/home/petalinux/
    ```
5. On the ZCU102, execute the following script **as root** to run the model:
    ``` bash
    cd target_zcu102/
    ./run_target
    ```
    You may have to make the file executable with `chmod +x run_model.sh`

    ---
    ***Nota bene**: if you don't want to build de C++ application each time, execute `./run_target --no-compile`, or modify the code.*

# Possible pitfalls :
- Failed to load xmodel subgraph: verify that there is only one subgraph. See [Run the Vitis AI flow](#Run-the-Vitis-AI-flow).
- Failed to build C++ app: log as root.
- Failed to launch a script: make the file executable with `chmod +x <script>`.
- If you don't have an ethernet connection, you can copy files directly on the SD card and connect to the ZCU102 with USB UART.
