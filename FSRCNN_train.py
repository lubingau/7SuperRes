import tensorflow as tf  
import numpy as np
import cv2
import json
import os
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from livelossplot import PlotLossesKeras
from sr_model import FSRCNN
from custom_metrics import PSNR

import argparse

def main(args):
    
    parameters_path = args.params_path
    with open(parameters_path, 'r') as f:
            training_parameters = json.load(f)

    gt_path = args.gt_path
    blr_path = args.blr_path

    labels = []
    features = []

    #Reading the patches, the dirs have to be sorted in order to match the feature to the right label

    for gt_patch_path in tqdm(sorted(os.listdir(gt_path), key=str.lower), desc="Loading high res patch"):
        gt_full_path =  os.path.join(gt_path, gt_patch_path)
        labels.append(cv2.imread(gt_full_path))

    y = np.stack(labels) / 255 #normalization
    y = y.astype(np.float16) #for a lower RAM footprint

    del labels

    for blr_patch_path in tqdm(sorted(os.listdir(blr_path), key=str.lower), desc="Loading blurred low res patch"):
        blr_full_path =  os.path.join(blr_path, blr_patch_path)
        features.append(cv2.imread(blr_full_path))

    x = np.stack(features)
    x = x / 255
    x = x.astype(np.float16)

    del features

    print("Dataset is ready. Features are of shape: ", x[0].shape, "Labels are of shape: ", y[0].shape)

    gc.collect()
    
    index = 20
    plt.figure()
    plt.subplot(121)
    plt.imshow(x[index].astype(np.float32))
    plt.axis('off')
    plt.title('Blured low res')
    plt.subplot(122)
    plt.imshow(y[index].astype(np.float32))
    plt.axis('off')
    plt.title('Orginal')
    plt.show() 
    
     #Splitting the dataset in train/val
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_parameters['test_ratio'], random_state=42)
    
    #Building the model
    BLR_IMG_SIZE = training_parameters['BLR_IMG_SIZE']
    UPSCALING_FACTOR = training_parameters['UPSCALING_FACTOR']
    COLOR_CHANNELS = training_parameters['COLOR_CHANNELS']

    model = FSRCNN(d=training_parameters['d'],s=training_parameters['s'],m=training_parameters['m'], 
                   input_size=BLR_IMG_SIZE, upscaling_factor=UPSCALING_FACTOR, color_channels=COLOR_CHANNELS)

    model.summary()
    
    model.compile(optimizer=RMSprop(learning_rate=training_parameters['lr']),
                  loss="MSE",
                  metrics=[PSNR])
    
    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=training_parameters['epochs'],
                        batch_size=training_parameters['batch_size'],
                        callbacks=[PlotLossesKeras()])
    
    model.save(training_parameters['model_name'] + '.h5')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', default=None, type=str, help='Path to high-resolution patches')
    parser.add_argument('--blr_path', default=None, type=str, help='Path to blurred low res patches')
    parser.add_argument('--params_path', default=None, type=str, help='Path to the training parameters json')

    args = parser.parse_args()
    
    main(args)
