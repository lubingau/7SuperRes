
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
    def __init__(self, imgs_dir, masks_dir, ratio=1, patch_size=512):
        self.image_patches = []
        self.mask_patches = []
        
        imgs_names = sorted(os.listdir(imgs_dir))
        masks_names = sorted(os.listdir(masks_dir))
        
        idx_choices = choices(range(0,len(imgs_names)), k=int(ratio*len(imgs_names)))
        
        toTens = torchvision.transforms.ToTensor()
        
        for idx in tqdm(idx_choices, desc='Loading imgs and masks'):
            filename = imgs_names[idx]
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = Image.open(os.path.join(imgs_dir, filename))
                image_patches = self._patchify(toTens(image), patch_size)
                self.image_patches.extend(image_patches)
            
            filename = masks_names[idx]
            if filename.endswith(".jpg") or filename.endswith(".png"):
                mask = Image.open(os.path.join(masks_dir, filename))
                mask = self.RGB_2_masks(mask)
                mask_patches = self._patchify(mask, patch_size)
                self.mask_patches.extend(mask_patches)
        
        # Convert the list of patches to a tensor
        self.image_patches = torch.stack(self.image_patches)
        self.mask_patches = torch.stack(self.mask_patches)
    
    def _patchify(self, image, patch_size):
        if(len(image.shape) == 3):
            width, height = image.shape[1], image.shape[2]
        else:
            width, height = image.shape[0], image.shape[1]
        patches = []
        for i in range(0, height-patch_size, patch_size):
            for j in range(0, width-patch_size, patch_size):
                if(len(image.shape) == 3):
                    patch = image[:, i:i+patch_size, j:j+patch_size]
                else:
                    patch = image[i:i+patch_size, j:j+patch_size]
                    
                patches.append(patch)
        return patches
    
    def RGB_2_masks(self, mask_to_be_converted):
        mapping = {(0  , 255, 255): 0,     #urban_land
                   (255, 255, 0  ): 1,    #agriculture
                   (255, 0  , 255): 2,    #rangeland
                   (0  , 255, 0  ): 3,      #forest_land
                   (0  , 0  , 255): 4,      #water
                   (255, 255, 255): 5,     #barren_land
                   (0  , 0  , 0  ): 6}     #unknown
        
        num_classes = 7
        
        temp = np.array(mask_to_be_converted)
        temp = np.where(temp>=128, 255, 0)

        class_mask=torch.from_numpy(temp)
        h, w = class_mask.shape[0], class_mask.shape[1]
        mask_tensor = torch.empty((num_classes, h, w), dtype=torch.bool)
        
        for i, k in enumerate(mapping):
            idx = torch.eq(class_mask, torch.tensor(k))
            validx = (idx.sum(2) == 3)
            mask_tensor[i] = validx
            
        return mask_tensor

    def __len__(self):
        return len(self.image_patches)
    
    def __getitem__(self, idx):
        return (self.image_patches[idx], self.mask_patches[idx])