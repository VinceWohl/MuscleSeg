# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:07:08 2023

@author: tp-vincentw

This script defiens the tansform functions.

"""

import torch
import numpy as np
import torchio as tio

np.random.seed(0)

class CropOrPadXY(tio.Transform):
    def __init__(self, target_shape):
        super().__init__()
        self.width = target_shape[0]
        self.height = target_shape[1]
        
    def apply_transform(self, subject):
              
        img = subject['image']
        mask = subject['mask']
        img_array = img.numpy()
        mask_array = mask.numpy()
        
        # X_dim
        x_dim = img_array.shape[1]
        if x_dim <= self.width:
            # pad x dim
            x_pad = self.width - x_dim
            xa = x_pad // 2
            xb = x_pad - xa
            img_array = np.pad(img_array, ((0, 0), (xa, xb), (0, 0), (0, 0)), mode='constant')
            mask_array = np.pad(mask_array, ((0, 0), (xa, xb), (0, 0), (0, 0)), mode='constant')
        else:
            # crop x dim
            start_x = (x_dim - self.width) // 2
            end_x = start_x + self.width
            img_array = img_array[:, start_x:end_x, :, :]
            mask_array = mask_array[:, start_x:end_x, :, :]
              
        # Y_dim
        y_dim = img_array.shape[2]
        if y_dim <= self.height:
            # pad y dim
            y_pad = self.height - y_dim
            ya = y_pad // 2
            yb = y_pad - ya
            img_array = np.pad(img_array, ((0, 0), (0, 0), (ya, yb), (0, 0)), mode='constant')
            mask_array = np.pad(mask_array, ((0, 0), (0, 0), (ya, yb), (0, 0)), mode='constant')
        else:
            # crop y dim
            start_y = (y_dim - self.height) // 2
            end_y = start_y + self.height
            img_array = img_array[:, :, start_y:end_y, :]
            mask_array = mask_array[:, :, start_y:end_y, :]
        
        img = torch.from_numpy(img_array)
        mask = torch.from_numpy(mask_array)
        subject['image'] = tio.Image(tensor=img, type=tio.INTENSITY)
        subject['mask'] = tio.Image(tensor=mask, type=tio.LABEL)
        
        return subject

class RandCropZ(tio.Transform):
    def __init__(self, target_shape):
        super().__init__()
        self.depth = target_shape[2]
        
    def apply_transform(self, subject):
        img = subject['image']
        mask = subject['mask']
        img_array = img.numpy()
        mask_array = mask.numpy()
        
        z_dim = img_array.shape[3]
        if z_dim > self.depth:
            # random cropping in depth lowest, middle, or highest window
            rNum = np.random.randint(3)
            
            if rNum == 0: # lowest
                seed = self.depth // 2
            elif rNum == 1: # middle
                seed = z_dim // 2
            elif rNum == 2: # highest
                seed = z_dim - self.depth // 2
            
            start_z = seed - self.depth // 2
            end_z = seed + self.depth // 2
            
            # crop z dimension
            cropped_img_array = img_array[:, :, :, start_z:end_z]
            cropped_mask_array = mask_array[:, :, :, start_z:end_z]
            
            img = torch.from_numpy(cropped_img_array)
            mask = torch.from_numpy(cropped_mask_array)
        else:
            # padding
            z = self.depth - z_dim
            za = z // 2
            zb = z - za
            
            # pad z dimension
            padded_img_array = np.pad(img_array, ((0, 0), (0, 0), (0, 0), (za, zb)), mode='constant')
            padded_mask_array = np.pad(mask_array, ((0, 0), (0, 0), (0, 0), (za, zb)), mode='constant')
            
            img = torch.from_numpy(padded_img_array)
            mask = torch.from_numpy(padded_mask_array)
        
        subject['image'] = tio.Image(tensor=img, type=tio.INTENSITY)
        subject['mask'] = tio.Image(tensor=mask, type=tio.LABEL)
            
        return subject

'''
For model A
'''
def get_transforms_A(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH):
    
    target_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH) # tio.CropOrPad expects dimensions (W, H, D)
    
    # transformation operator for TRAINING data
    train_transform = tio.Compose([
        CropOrPadXY(target_shape),
        RandCropZ(target_shape),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    # transformation operator for VALIDATION data
    val_transform = tio.Compose([
        CropOrPadXY(target_shape),
        RandCropZ(target_shape),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    # transformation operator for TESTING data
    test_transform = tio.Compose([
        CropOrPadXY(target_shape),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    return train_transform, val_transform, test_transform


'''
For model B
'''
def get_transforms_B(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH):
    
    target_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH) # tio.CropOrPad expects dimensions (W, H, D)
    
    augmentation = {
        tio.RandomBiasField(coefficients=0.3) : 0.5,
        tio.RandomElasticDeformation(num_control_points=8, max_displacement=2, locked_borders=2): 0.5,
    }
    
    # transformation operator for TRAINING data
    train_transform = tio.Compose([
        CropOrPadXY(target_shape),
        RandCropZ(target_shape),
        tio.OneOf(augmentation, p=0.75),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    # transformation operator for VALIDATION data
    val_transform = tio.Compose([
        CropOrPadXY(target_shape),
        RandCropZ(target_shape),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    # transformation operator for TESTING data
    test_transform = tio.Compose([
        CropOrPadXY(target_shape),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    return train_transform, val_transform, test_transform


'''
For model C
'''
def get_transforms_C(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH):
    
    target_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH) # tio.CropOrPad expects dimensions (W, H, D)
    
    augmentation = {
        tio.RandomBiasField(coefficients=0.3) : 0.5,
        tio.RandomElasticDeformation(num_control_points=8, max_displacement=2, locked_borders=2): 0.5,
    }
    
    # transformation operator for TRAINING data
    train_transform = tio.Compose([
        CropOrPadXY(target_shape),
        RandCropZ(target_shape),
        tio.OneOf(augmentation),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    # transformation operator for VALIDATION data
    val_transform = tio.Compose([
        CropOrPadXY(target_shape),
        RandCropZ(target_shape),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    # transformation operator for TESTING data
    test_transform = tio.Compose([
        CropOrPadXY(target_shape),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    
    return train_transform, val_transform, test_transform