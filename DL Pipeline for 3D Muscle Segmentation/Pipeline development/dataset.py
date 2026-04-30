# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:18:25 2023

@author: tp-vincentw

This script defines the MRI dataset.

"""

import torch
import torchio as tio
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset


class MRI_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes, transform=None):
        self.FirCImgs = image_dir[0][:]
        self.SecCImgs = image_dir[1][:]
        self.masks = mask_dir
        self.transform = transform
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.masks)
    
    def read_nifti_file(self, path):
        nifti = nib.load(path)
        header = nifti.header.copy()
        array = nifti.get_fdata()
        tensor = torch.from_numpy(array)
        return tensor, header
    
    def get_headers(self, index):
        _, header_img = self.read_nifti_file(self.FirCImgs[index])
        _, header_mask = self.read_nifti_file(self.masks[index])
        return header_img, header_mask
    
    def preprocess(self, img, mask):
        # data preprocessing and augmentation is applied with transform
        if self.transform is not None:
            
            # Create a TorchIO Subject
            subject = tio.Subject(
                image = tio.Image(tensor=img),          # torchio Image
                mask = tio.ScalarImage(tensor=mask)     # torchio Mask
                )
            
            # Apply transformation
            subject_transformed = self.transform(subject)
            
            # Reconvert to image and mask tensor
            img = subject_transformed["image"].tensor
            mask = subject_transformed["mask"].tensor
        
        return img, mask
        
    def __getitem__(self, index):
        
        # first channel image tensor
        firC,_ = self.read_nifti_file(self.FirCImgs[index])
        
        # second channel image tensor
        secC,_ = self.read_nifti_file(self.SecCImgs[index])
        
        # concatenate first and second image channel to one image tensor with two channels
        img = torch.stack([firC, secC], dim=0) # image shape(C, W, H, D)
        
        # mask tensor
        mask,_ = self.read_nifti_file(self.masks[index])       # mask shape (W, H, D)
        
        #preprocess the data
        mask = mask.unsqueeze(0)
        img, mask = self.preprocess(img, mask)
        mask = mask.squeeze(0)
        
        # convert the mask tensor to categorical representation
        mask = F.one_hot(mask.long(), self.num_classes).permute(3, 0, 1, 2)  # mask shape (num_classes, W, H, D)
            
        img = img.to(torch.float32)     # tensor(C, W, H, D)
        mask = mask.to(torch.float32)   # tensor(N, W, H, D)
        
        return img, mask