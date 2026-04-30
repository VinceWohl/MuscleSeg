# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:21:50 2023

@author: tp-vincentw

This script applies cropping on the MR images by taking the SAT masks to define a bounding box.
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label
import torch.nn.functional as F
from utils import (read_nifti_file)


def get_files(main_folder, app_folder):
    FirCImgs = []
    SecCImgs = []
    ThiCImgs = []
    FouCImgs = []
    WholeMasks = []
    CompMasks = []
    
    # get original images and muscle compartment masks from the main data folder
    subjects = os.listdir(main_folder)
    subjects = sorted(subjects, key=lambda x: int(x))
    
    for subject in subjects:
        subject_path = main_folder + '/' + subject
                
        parts = os.listdir(subject_path)
        for part in parts:
            part_path = subject_path + '/' + part
            
            FirCImgs.append(part_path + '/' + 'roi_70p_Opp_Phase.nii.gz')
            SecCImgs.append(part_path + '/' + 'roi_70p_In_Phase.nii.gz')
            ThiCImgs.append(part_path + '/' + 'roi_70p_Water.nii.gz')
            FouCImgs.append(part_path + '/' + 'roi_70p_Fat.nii.gz')
            
            if part == 'CALF':
                CompMasks.append(part_path + '/' + subject + '_CALF_MUSCLE_COMP_mask.nii.gz')
            if part == 'THIGH':
                CompMasks.append(part_path + '/' + subject + '_THIGH_MUSCLE_COMP_mask.nii.gz')
    
    # get whole muscle and SAT masks out of the application folder
    files = os.listdir(app_folder)
    
    for file in files:
        if 'WHOLEMUSCLE_SAT_pred.nii.gz' in file:
            WholeMasks.append(f'{app_folder}/{file}')
    WholeMasks = sorted(WholeMasks)
    
    return FirCImgs, SecCImgs, ThiCImgs, FouCImgs, WholeMasks, CompMasks


def find_bbox(mask_array):

    wholemuscle = mask_array[2, :, :, :]
    
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


def BRIDGE(data_path, app_folder):
    # get all the images and the corresponding whole muscle & SAT masks
    FirCImgs, SecCImgs, ThiCImgs, FouCImgs, WholeMasks, CompMasks = get_files(data_path, app_folder)
    
    for idx in range(len(WholeMasks)):
        
        # get the files dirctions
        opp_dir = FirCImgs[idx]
        in_dir = SecCImgs[idx]
        water_dir = ThiCImgs[idx]
        fat_dir = FouCImgs[idx]
        whole_mask_dir = WholeMasks[idx]
        comp_mask_dir = CompMasks[idx]
        
        # read the images and masks
        oppPhase, header_opp = read_nifti_file(opp_dir)
        inPhase, header_in = read_nifti_file(in_dir)
        water, header_water = read_nifti_file(water_dir)
        fat, header_fat = read_nifti_file(fat_dir)
        whole_mask, header_wm = read_nifti_file(whole_mask_dir)
        comp_mask, header_cm = read_nifti_file(comp_mask_dir)
        
        # convert the whole mask tensor to categorical representation
        class_mask = F.one_hot(whole_mask.long(), num_classes=3).permute(3, 0, 1, 2) # mask shape (num_classes, W, H, D)
        class_mask = class_mask.numpy()
        
        # convert tensors to arrays
        opp_array = oppPhase.numpy()
        in_array = inPhase.numpy()
        water_array = water.numpy()
        fat_array = fat.numpy()
        whole_mask_array = whole_mask.numpy().astype(int)
        comp_mask_array = comp_mask.numpy().astype(int)
                
        # find the ideal bounding box for the cropping 
        min_x, max_x, min_y, max_y = find_bbox(class_mask)
            
        # crop images, and muscle compartment mask all with 3 dims
        opp_array = opp_array[min_x:max_x, min_y:max_y, :]
        in_array = in_array[min_x:max_x, min_y:max_y, :]
        water_array = water_array[min_x:max_x, min_y:max_y, :]
        fat_array = fat_array[min_x:max_x, min_y:max_y, :]
        whole_mask_array = whole_mask_array[min_x:max_x, min_y:max_y, :]
        comp_mask_array = comp_mask_array[min_x:max_x, min_y:max_y, :]
        
        # convert to nifti
        opp_nifti = nib.Nifti1Image(opp_array, affine=None, header=header_opp, dtype=np.uint16)  
        in_nifti = nib.Nifti1Image(in_array, affine=None, header=header_in, dtype=np.uint16)
        water_nifti = nib.Nifti1Image(water_array, affine=None, header=header_in, dtype=np.uint16)
        fat_nifti = nib.Nifti1Image(fat_array, affine=None, header=header_in, dtype=np.uint16)
        whole_mask_nifti = nib.Nifti1Image(whole_mask_array, affine=None, header=header_wm, dtype=np.uint16)
        comp_mask_nifti = nib.Nifti1Image(comp_mask_array, affine=None, header=header_cm, dtype=np.uint16)
        
        # safe the files in app_folder instead of data_path
        opp_name = comp_mask_dir.split('/')[-1].replace('MUSCLE_COMP_mask.nii.gz', 'cropped_roi_70p_Opp_Phase.nii.gz')
        in_name = comp_mask_dir.split('/')[-1].replace('MUSCLE_COMP_mask.nii.gz', 'cropped_roi_70p_In_Phase.nii.gz')
        water_name = comp_mask_dir.split('/')[-1].replace('MUSCLE_COMP_mask.nii.gz', 'cropped_roi_70p_Water.nii.gz')
        fat_name = comp_mask_dir.split('/')[-1].replace('MUSCLE_COMP_mask.nii.gz', 'cropped_roi_70p_Fat.nii.gz')
        whole_mask_name = whole_mask_dir.split('/')[-1].replace('WHOLEMUSCLE_SAT_pred.nii.gz', 'WHOLEMUSCLE_SAT_cropped_pred.nii.gz')
        comp_mask_name = comp_mask_dir.split('/')[-1].replace('MUSCLE_COMP_mask.nii.gz', 'MUSCLE_COMP_cropped_mask.nii.gz')
        
        # # save cropped images and masks
        nib.save(opp_nifti, f'{app_folder}/{opp_name}')
        nib.save(in_nifti, f'{app_folder}/{in_name}')
        nib.save(water_nifti, f'{app_folder}/{water_name}')
        nib.save(fat_nifti, f'{app_folder}/{fat_name}')
        nib.save(whole_mask_nifti, f'{app_folder}/{whole_mask_name}')
        nib.save(comp_mask_nifti, f'{app_folder}/{comp_mask_name}')
        
        # # manage file names to keep the original ones
        # opp_dir = opp_dir.replace('roi_70p_Opp_Phase.nii.gz', 'cropped_roi_70p_Opp_Phase.nii.gz')
        # in_dir = in_dir.replace('roi_70p_In_Phase.nii.gz', 'cropped_roi_70p_In_Phase.nii.gz')
        # water_dir = water_dir.replace('roi_70p_Water.nii.gz', 'cropped_roi_70p_Water.nii.gz')
        # fat_dir = fat_dir.replace('roi_70p_Fat.nii.gz', 'cropped_roi_70p_Fat.nii.gz')
        # whole_mask_dir = whole_mask_dir.replace('WHOLEMUSCLE_SAT_mask.nii.gz', 'WHOLEMUSCLE_SAT_cropped_mask.nii.gz')
        # comp_mask_dir = comp_mask_dir.replace('MUSCLE_COMP_mask.nii.gz', 'MUSCLE_COMP_cropped_mask.nii.gz')
    
        # # save cropped images and masks
        # nib.save(opp_nifti, opp_dir)
        # nib.save(in_nifti, in_dir)
        # nib.save(water_nifti, water_dir)
        # nib.save(fat_nifti, fat_dir)
        # nib.save(whole_mask_nifti, whole_mask_dir)
        # nib.save(comp_mask_nifti, comp_mask_dir)
        
        print(f'=> saved cropped versions of OP, IP, W, F, whole muscle and compartments masks: {comp_mask_dir}') 