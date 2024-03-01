#!/usr/bin/env python

import os
import torch
from PIL import Image


class SuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, dir_blr, dir_gt, calib_num_img, transform=None):
        self.dir_blr = dir_blr
        self.dir_gt = dir_gt
        self.transform = transform
        self.blr_filenames = os.listdir(dir_blr)
        self.gt_filenames = os.listdir(dir_gt)
        
        self.blr_filenames.sort()
        self.blr_filenames = self.blr_filenames[:calib_num_img]
        self.gt_filenames.sort()
        self.gt_filenames = self.gt_filenames[:calib_num_img]

    def __len__(self):
        return len(self.blr_filenames)

    def __getitem__(self, idx):
        blr_img_name = os.path.join(self.dir_blr, self.blr_filenames[idx])
        gt_img_name = os.path.join(self.dir_gt, self.gt_filenames[idx])
        
        blr_image = Image.open(blr_img_name)
        gt_image = Image.open(gt_img_name)
        
        if self.transform:
            blr_image = self.transform(blr_image)
            gt_image = self.transform(gt_image)
        
        return blr_image, gt_image
