# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:18:25 2023

@author: tp-vincentw

This script defines the MRI dataset.

"""

import torch
import torchio as tio
import nibabel as nib
from torch.utils.data import Dataset


class MRI_Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.FirCImgs = image_dir[0][:]
        self.SecCImgs = image_dir[1][:]
        self.transform = transform
        
    def __len__(self):
        return len(self.FirCImgs)
    
    def read_nifti_file(self, path):
        nifti = nib.load(path)
        header = nifti.header.copy()
        array = nifti.get_fdata()
        tensor = torch.from_numpy(array)
        return tensor, header
    
    def get_headers(self, index):
        _, header_img = self.read_nifti_file(self.FirCImgs[index])
        return header_img
    
    def get_init_WnH(self, index):
        nifti = nib.load(self.FirCImgs[index])
        array = nifti.get_fdata()
        return [array.shape[0], array.shape[1]]
    
    def preprocess(self, img):
        # data preprocessing and augmentation is applied with transform
        if self.transform is not None:
            # Create a TorchIO Subject
            subject = tio.Subject(image = tio.Image(tensor=img))
            # Apply transformation
            subject_transformed = self.transform(subject)
        
        return subject_transformed['image'].tensor
        
    def __getitem__(self, index):
        
        # first channel image tensor
        firC,_ = self.read_nifti_file(self.FirCImgs[index])
        
        # second channel image tensor
        secC,_ = self.read_nifti_file(self.SecCImgs[index])
        
        # concatenate first and second image channel to one image tensor with two channels
        img = torch.stack([firC, secC], dim=0) # image shape(C, W, H, D)
        
        #preprocess the data
        img = self.preprocess(img)
            
        img = img.to(torch.float32) # tensor(C, W, H, D)
        
        return img