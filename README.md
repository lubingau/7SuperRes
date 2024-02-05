# DeepZoom : a super resolution AI on ZCU102


Project presentation

# How to train the model

Scrip to train the model :
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

Here is a script that create a dataset with this 


# How to compile the model with Vitis-AI

## Requirements
- Docker
- Linux host PC
- ... ... ...

## Installation
1. Download the [Repo](https://bit.ly/pynqz1_2_7)
2. Install Vitis-AI on your PC
    ```bash
    ...
    ```
3. ...
4. Clone or copy this repository on the PYNQ.

## Run the Vitis-AI flow
1. Launch the docker
2. Launch the flow :

    ```bash
    ...
    ```
3. Output files will be in `directory`



# How to install and run the model on ZCU102

1. Export files on the ZCU102 :

    ```bash
    scp files zcu-102:/home/petalinux
    ```
2. Run the script

# How to train a model, convert it to onnx, then to tensil and finally run it on the PYNQ
## Schema of the process
![plot](./static/process.png)




# Possible pitfalls :
    - ...
    - ...
    - ...