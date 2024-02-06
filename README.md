![Logo DeepZoom](./doc/logo.png)

# DeepZoom : a super resolution AI on ZCU102

The DeepZoom project is an initiative by Airbus Defense and Space, proposed to students at ENSEEIHT as part of an Industrial Study Project (BEI). The objective of this project is to develop an embedded application for satellite image super-resolution using AI techniques. Ultimately, the application should be capable of being deployed on an already-orbiting satellite. The system is a sandbox (thus reconfigurable) and is equipped with an optical sensor capable of capturing RGB images with a resolution of 150 million pixels (14192x10640).


# How to train the model

Téléchargez notre dataset ici (extrait de [Cars Overhead With Context](https://gdo152.llnl.gov/cowc/)). Notre modèle est un FSRCNN adapté de "*[Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367)*" dont les paramètres sont les suivants :


Pour entrainer notre modèle, veuillez exécuter ce script python :

Script to train the model :
```bash
python3 train.py --dataset path-to-dataset
```

Here are all options :

```bash
options:
  -h, --help            show this help message and exit
  --dataset PATH
                        path the dataset
  ... ... ... ...

```
## How to train with your own dataset

Nous proposons un script python qui vous permet de créer votre dataset avec vos propre images:
```bash
cd dataset/
python3 generate_dataset.py --images path-to-your-images
```

# How to compile the model with Vitis-AI

Une fois que le modèle est entrainé, il faut d'abord le quantizer et le compiler avec les outils de Vitis AI. Nous utilisons la version 3.0. Voici un schéma représentant notre flow Vitis AI:

![Flow Vitis AI](./doc/flow-vitis-ai.png)

## Requirements

- Ubuntu 22.04 host PC
- Vitis AI 3.0 repository
- Docker

## Installation
1. Clone or download the [Vitis AI 3.0](https://github.com/Xilinx/Vitis-AI/tree/3.0) repository. ***WARNING** : be careful to use the 3.0 version*.
    ```bash
    git clone https://github.com/Xilinx/Vitis-AI
    git checkout -b 3.0 origin/3.0
    ```
2. Install Vitis-AI on your PC
    ```bash
    cd Vitis-AI/
    cd docker/
    sudo ./docker_build.sh -t cpu -f tf2
    ```
    This operation might take some time (~20 minutes). If you want to use gpu instead of cpu, you can but we haven't tried it
3. Once the process is finished, with the command ```sudo docker images``` you should see something like this:
    ```text
    REPOSITORY                        TAG      IMAGE ID       CREATED         SIZE
    xilinx/vitis-ai-tensorflow2-cpu   latest   e1501ac96fd0   10 days ago     6.75GB
    ```
4. You will need to install some missing packages and libraries into the Vitis AI container. Copy the file `setup_docker_env.sh` into the `Vitis-AI` folder
    ```bash
    cd DeepZoom
    cp setup_docker_env.sh Vitis-AI/
    cp -r pkgs/ Vitis-AI/src/vai_quantizer/vai_q_tensorflow2.x/
    ```
5. Copy the `dz-flow`

5. To launch the docker container with Vitis AI tools, execute the following commands from the `Vitis-AI` folder:
    ```bash
    cd Vitis-AI/
    sudo ./docker_run.sh xilinx/vitis-ai-tensorflow2-cpu:latest
    ```
6. Once you are into the Vitis AI container, execute the following script to install some missing packages and libraries:
    ```bash
    sudo ./setup_docker_env.sh
    ```
6. Voilà! You are ready to play with Vitis AI tools.

***WARNING**: you will need to execute the `setup_docker_env.sh` script each time you launch Vitis AI container.*

## Run the Vitis-AI flow
1. Launch the Vitis AI container. Be careful to execute `setup_docker_env.sh` script.
2. Launch the flow :
    ```bash
    cd flow-vitis-ai/
    ./run_model.sh
    ```
3. Output files will be in `build/`:
    - `0_log`: all log
    - `1_quantize_model`: quantize model
    - `2_predictions`: all predictions (png files)
    - `3_compile_model`: output compiled model and subgraphs
4. Output files will be automatically copy into `target_zcu102` folder

*Vous pouvez vérifier que la compilation est réussie en visualisant le fichier `subgraph_model.png`. Ce fichier représente le contenu du fichier `.xmodel` qui contient toutes les informations nécessaires au DPU. Le fichier sera executable si et seulement si il n'y qu'**un seul bloc DPU** (bleu). Dans le cas ou il existe plusieurs blocs DPU, cela signifie que des opérations sont réalisées sur le CPU au sein du modèle. Cela peut-être dû à une opération mathématique non supportée par le DPU, une fonction d'activation par exemple. Vérifiez les logs Vitis AI ainsi que la [documentation Vitis AI](https://docs.xilinx.com/r/3.0-English/ug1414-vitis-ai/Currently-Supported-Operators).*

## Compiled your model

Il vous est possible de quantizer et compiler votre propre modèle. Pour cela, veuillez copier votre modèle entrainé en tensorflow2 (dernière version de tensorflow) dans le dossier `input_model`. Puis exécuter le flow Vitis AI en lui précisant le nom de votre modèle. Le flow s'occupera de créer un dossier pour ce modèle pour toutes les étapes clées du flow:

```bash
./run_model your_model.h5
```

## Run on the ZCU102

1. Flash [ZCU102 DPU image](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu102-dpu-v2022.2-v3.0.0.img.gz) on the SD card with balenaEtcher for example. Then, insert the SD card into the ZCU102, and switch on the board. Be careful to put correctly boot mode switches on the ZCU102 : 1000.

2. When you power on the ZCU102, a red led appear, and then disappear when boot is done. You can connect to the board with UART or ethernet. Prefer the ethernet connection for file transfer, otherwise you will have to remove the SD card.

3. Copy the `target_zcu102` folder from the host PC to the ZCU102:
```bash
cd Vitis-AI/flow-vitis-ai/
scp -r target_zcu102 root@192.168.1.64:home/petalinux/
```

4. On the ZCU102, execute the following script **as root** to run the model:
``` bash
cd target_zcu102/
./run_target
```
***Nota bene 1**: if you want to run your model, execute `./run_target your-model`, or modify the code.*

***Nota bene 2**: if you don't want to build de C++ application each time, execute `./run_target --no-compile`, or modify the code.*

# Possible pitfalls :
- Failed to load xmodel subgraph: verify that there is only one subgraph. See [Run the Vitis AI flow](#Run-the-Vitis-AI-flow).
- Failed to build C++ app: log as a root.
