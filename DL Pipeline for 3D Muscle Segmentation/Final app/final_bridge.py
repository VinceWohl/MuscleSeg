# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:21:50 2023

@author: tp-vincentw

This script applies crops the MR images around the whole muscle region.
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label
import torch.nn.functional as F
from final_utils import read_nifti_file


def get_files(main_folder):     # for NIMMI data folder
    FirCImgs = []
    SecCImgs = []
    ThiCImgs = []
    FouCImgs = []
    WholeMasks = []
    
    subjects = sorted(os.listdir(main_folder))
    for subject in subjects:
        subject_folder = main_folder + '/' + subject
        visits = sorted(os.listdir(subject_folder))
        for visit in visits:
            visit_folder = subject_folder + '/' + visit
            parts = sorted(os.listdir(visit_folder))
            for part in parts:
                part_folder = visit_folder + '/' + part
                scans = sorted(os.listdir(part_folder))
                for scan in scans:
                    if 'roi_70p' in scan and 'F' in scan:
                        FirCImgs.append(part_folder + '/' + scan)
                    if 'roi_70p' in scan and 'in' in scan:
                        SecCImgs.append(part_folder + '/' + scan)
                    if 'roi_70p' in scan and 'opp' in scan:
                        ThiCImgs.append(part_folder + '/' + scan)
                    if 'roi_70p' in scan and 'W' in scan:
                        FouCImgs.append(part_folder + '/' + scan)
                    if 'WHOLEMUSCLE_SAT_mask' in scan:
                        WholeMasks.append(part_folder + '/' + scan)
    return FirCImgs, SecCImgs, ThiCImgs, FouCImgs, WholeMasks


def find_bbox(wholemuscle):

    labels, num_labels = label(wholemuscle)
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1  # Skip background label 0
    largest_contour_mask = labels == largest_label
    
    indices = np.where(largest_contour_mask)
    min_x, max_x = np.min(indices[0]), np.max(indices[0])
    min_y, max_y = np.min(indices[1]), np.max(indices[1])
    
    min_x -= 10
    max_x += 10
    min_y -= 10
    max_y += 10

    return min_x, max_x, min_y, max_y


def BRIDGE(main_folder):
    FirCImgs, SecCImgs, ThiCImgs, FouCImgs, WholeMasks = get_files(main_folder)
    shape_diffs = {'WholeMuscle masks': [], 'buffer': []}
    
    for idx in range(len(WholeMasks)):
        
        # read the images and masks
        fat, header_fat = read_nifti_file(FirCImgs[idx])
        inPhase, header_in = read_nifti_file(SecCImgs[idx])
        oppPhase, header_opp = read_nifti_file(ThiCImgs[idx])
        water, header_water = read_nifti_file(FouCImgs[idx])
        whole_mask, header_wm = read_nifti_file(WholeMasks[idx])
        print(f'Cropping {FirCImgs[idx]} based on {WholeMasks[idx]}')
        
        # convert the whole mask tensor to categorical representation and pull out the whole muscle segmentation
        class_mask = F.one_hot(whole_mask.long(), num_classes=3).permute(3, 0, 1, 2) # mask shape (num_classes, W, H, D)
        class_mask = class_mask.numpy()
        wholemuscle = class_mask[2, :, :, :]
        
        # convert tensors to arrays
        fat_array = fat.numpy()
        in_array = inPhase.numpy()
        opp_array = oppPhase.numpy()
        water_array = water.numpy()
                
        # find the ideal bounding box for the cropping 
        min_x, max_x, min_y, max_y = find_bbox(wholemuscle)
        
        # save shape diffs for later
        shape_diffs['WholeMuscle masks'].append(WholeMasks[idx])
        shape_diffs['buffer'].append([min_x, fat_array.shape[0]-max_x, min_y, fat_array.shape[1]-max_y])
            
        # crop images, and muscle compartment mask all with 3 dims
        fat_array = fat_array[min_x:max_x, min_y:max_y, :]
        in_array = in_array[min_x:max_x, min_y:max_y, :]
        opp_array = opp_array[min_x:max_x, min_y:max_y, :]
        water_array = water_array[min_x:max_x, min_y:max_y, :]
        
        # convert to nifti
        fat_nifti = nib.Nifti1Image(fat_array, affine=None, header=header_in, dtype=np.uint16)
        in_nifti = nib.Nifti1Image(in_array, affine=None, header=header_in, dtype=np.uint16)
        opp_nifti = nib.Nifti1Image(opp_array, affine=None, header=header_opp, dtype=np.uint16)  
        water_nifti = nib.Nifti1Image(water_array, affine=None, header=header_in, dtype=np.uint16)
        
        # safe the files in app_folder instead of data_path
        fat_name = FirCImgs[idx].replace('roi_70p', 'roi_cropped_70p')
        in_name = SecCImgs[idx].replace('roi_70p', 'roi_cropped_70p')
        opp_name = ThiCImgs[idx].replace('roi_70p', 'roi_cropped_70p')
        water_name = FouCImgs[idx].replace('roi_70p', 'roi_cropped_70p')
        
        # # save cropped images and masks
        nib.save(fat_nifti, fat_name)
        nib.save(in_nifti, in_name)
        nib.save(opp_nifti, opp_name)
        nib.save(water_nifti, water_name)
   
    print('=> saved cropped versions of F, W, In, Opp scans and WHOLEMUSLCE_SAT_cropped_mask')
    
    return shape_diffs