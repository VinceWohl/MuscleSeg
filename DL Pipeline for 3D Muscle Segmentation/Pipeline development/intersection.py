# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:37:08 2023

@author: tp-vincentw

This script generates MAT and IMAT masks by overlaying the fat images on the whole muscle and muscle compartment masks.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from utils import(read_nifti_file)


def get_files(folder):
    FatImgs = []
    WholeMasks = []
    CompMasks = []
    
    files = os.listdir(folder)          
    for file in files:
        if 'cropped_roi_70p_Fat.nii.gz' in file:
            FatImgs.append(f'{folder}/{file}')
        if 'WHOLEMUSCLE_SAT_cropped_pred.nii.gz' in file:
            WholeMasks.append(f'{folder}/{file}')
        if 'MUSCLE_COMP_cropped_pred.nii.gz' in file:
            CompMasks.append(f'{folder}/{file}')
    
    FatImgs = sorted(FatImgs)
    WholeMasks = sorted(WholeMasks)
    CompMasks = sorted(CompMasks)
    
    return FatImgs, WholeMasks, CompMasks


def INTERSECTION(app_folder, FIND_IMAT=False):

    FatImgs, WholeMasks, CompMasks = get_files(app_folder)
    
    # create a volumes matrix to save the MAT, Intra- and InterMAT volumes for each sample
    v_matrix = []
    head_row = ['Fat image', 'V(MAT)', 'V(IntraMAT)', 'V(InterMAT)']
    v_matrix.append(head_row)
    
    for idx in range(len(FatImgs)):
        '''
        ##################################################################################################################
        quantify MAT
        '''
        # prepare fat image
        fat, _ = read_nifti_file(FatImgs[idx])
        fat = fat.numpy()
        fat = (fat - np.amin(fat)) / (np.amax(fat) - np.amin(fat))  # normalize voxel intensities between 0 and 1
        fat = np.where(fat < 0.1, 0, 1)                             # thresholding for voxel intensities
        
        # prepare wholemuscle mask
        mask, header = read_nifti_file(WholeMasks[idx])
        mask = F.one_hot(mask.long(), num_classes=3).permute(3, 0, 1, 2)  # mask shape (num_classes, W, H, D)
        wholemuscle = mask[2, :, :, :]
        wholemuscle = wholemuscle.numpy()
        
        # find intersection
        MAT = fat * wholemuscle
                
        # save MAT mask
        MAT_nifti = nib.Nifti1Image(MAT, affine=None, header=header, dtype=np.uint16)
        MAT_dir = WholeMasks[idx].replace('WHOLEMUSCLE_SAT_cropped_pred.nii.gz', 'MAT_cropped_pred.nii.gz')
        nib.save(MAT_nifti, MAT_dir)
        print(f'=> saved MAT mask: {MAT_dir}')
        
        # save MAT volume in volumes matrix
        new_row = [FatImgs[idx], (MAT.sum())*header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3]]
    
    
        if FIND_IMAT:
            '''
            ##################################################################################################################
            quantify IntraMAT
            '''
            # prepare muscle compartment mask
            mask, header = read_nifti_file(CompMasks[idx])
            mask = mask.numpy()
            
            # find intersection
            IntraMAT = fat * mask
            
            # save IntraMAT mask
            IntraMAT_nifti = nib.Nifti1Image(IntraMAT, affine=None, header=header, dtype=np.uint16)
            IntraMAT_dir = CompMasks[idx].replace('MUSCLE_COMP_cropped_pred.nii.gz', 'IntraMAT_COMP_cropped_pred.nii.gz')
            nib.save(IntraMAT_nifti, IntraMAT_dir)
            print(f'=> saved IntraMAT mask: {IntraMAT_dir}')
            
            # save IntraMAT volume in volumes matrix
            IntraMAT[IntraMAT > 0] = 1
            new_row.append((IntraMAT.sum())*header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3])
            
            '''
            ##################################################################################################################
            quantify InterMAT
            '''
            # find intersection (bit more complex: there are edge regions which can be seen as IntraMAT but not as MAT)
            IntraMAT[IntraMAT > 0] = 2
            InterMAT = MAT - IntraMAT
            InterMAT[InterMAT < 0] = 0
            InterMAT[InterMAT > 0] = 1
            
            IMAT = InterMAT + IntraMAT
            
            # save IMAT mask (IMAT = InterMAT + IntraMAT)
            IMAT_nifti = nib.Nifti1Image(IMAT, affine=None, header=header, dtype=np.uint16)
            IMAT_dir = CompMasks[idx].replace('MUSCLE_COMP_cropped_pred.nii.gz', 'IMAT_pred.nii.gz')
            nib.save(IMAT_nifti, IMAT_dir)
            print(f'=> saved IMAT mask: {IMAT_dir}')
            
            # save InterMAT volumes in volumes matrix
            new_row.append((InterMAT.sum())*header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3])
            
        v_matrix.append(new_row)
      
    # save volumes matrix in excel sheet
    v_matrix_df = pd.DataFrame(v_matrix[1:], columns=v_matrix[0])
    file_path = f"{app_folder}/MAT_volumes.xlsx"
    writer = pd.ExcelWriter(file_path, engine="xlsxwriter") 
    print("=> Saving MAT volumes")
    v_matrix_df.to_excel(writer, sheet_name="MAT volumes", index=False)
    writer.close()