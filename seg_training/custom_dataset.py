import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from random import choices

class SegmentationDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, ratio=1, patch_size=256):
        self.num_classes = 7

        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        
        self.patch_size = patch_size
        
        list_imgs_dir = sorted(os.listdir(imgs_dir))
        list_masks_dir = sorted(os.listdir(masks_dir))
        self.imgs_names = [f for f in list_imgs_dir[:int(ratio*len(list_imgs_dir))] if f.endswith(('.jpg', '.png'))]
        self.masks_names = [f for f in list_masks_dir[:int(ratio*len(list_masks_dir))] if f.endswith(('.jpg', '.png'))]
        self.dataset_len = len(self.imgs_names)
        
        self.toTens = torchvision.transforms.ToTensor()
        
        
    def _patchify(self, image, patch_size):
        width, height = image.shape[1], image.shape[2]
        patches = []
        for i in range(0, height-patch_size, patch_size):
            for j in range(0, width-patch_size, patch_size):
                patch = image[:, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        return torch.stack(patches)
    
    def RGB_2_masks(self, mask_to_be_converted):
        mapping = {(0  , 255, 255): 0,     #urban_land
                   (255, 255, 0  ): 1,    #agriculture
                   (255, 0  , 255): 2,    #rangeland
                   (0  , 255, 0  ): 3,      #forest_land
                   (0  , 0  , 255): 4,      #water
                   (255, 255, 255): 5,     #barren_land
                   (0  , 0  , 0  ): 6}     #unknown
        
        temp = np.array(mask_to_be_converted)
        temp = np.where(temp>=128, 255, 0)

        class_mask=torch.from_numpy(temp)
        h, w = class_mask.shape[0], class_mask.shape[1]
        mask_tensor = torch.empty((self.num_classes, h, w), dtype=torch.bool)
        
        for i, k in enumerate(mapping):
            idx = torch.eq(class_mask, torch.tensor(k))
            validx = (idx.sum(2) == 3)
            mask_tensor[i] = validx
            
        return mask_tensor

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.imgs_dir, self.imgs_names[idx]))
        image = self.toTens(image)
        mask = Image.open(os.path.join(self.masks_dir, self.masks_names[idx]))
        mask = self.RGB_2_masks(mask)
        return (image, mask)

