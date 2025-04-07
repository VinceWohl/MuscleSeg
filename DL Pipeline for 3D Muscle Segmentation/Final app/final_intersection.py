# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:37:08 2023

@author: tp-vincentw

This script generates MAT and IMAT masks by overlaying the fat images on the whole muscle and muscle compartment masks and quantifies the adipose tissues.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from final_utils import read_nifti_file


def get_files(main_folder):         # for NIMMI data folder
    FatImgs = []
    WholeMasks = []
    ThighCompMasks = []
    CalfCompMasks = []
    
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
                    if scan == 'roi_70p_thigh_F.nii.gz' or scan == 'roi_70p_calf_F.nii.gz':
                        FatImgs.append(part_folder + '/' + scan)
                    if scan == 'WHOLEMUSCLE_SAT_mask.nii.gz':
                        WholeMasks.append(part_folder + '/' + scan)
                    if part == 'THIGH' and scan == 'MUSCLECOMP_mask.nii.gz':
                        ThighCompMasks.append(part_folder + '/' + scan)
                    if part == 'CALF' and scan == 'MUSCLECOMP_mask.nii.gz':
                        CalfCompMasks.append(part_folder + '/' + scan)                       
                    
    return FatImgs, WholeMasks, ThighCompMasks, CalfCompMasks



def get_files_tmp(main_folder):         # for NIMMI data folder
    FatImgs = []
    WholeMasks = []
    ThighCompMasks = []
    CalfCompMasks = []
    
    subjects = sorted(os.listdir(main_folder))
    for subject in subjects:
        subject_folder = main_folder + '/' + subject
        parts = sorted(os.listdir(subject_folder))
        for part in parts:
            part_folder = subject_folder + '/' + part
            scans = sorted(os.listdir(part_folder))
            for scan in scans:
                if scan == 'roi_70p_thigh_F.nii.gz' or scan == 'roi_70p_calf_F.nii.gz':
                    FatImgs.append(part_folder + '/' + scan)
                if scan == 'WHOLEMUSCLE_SAT_mask.nii.gz':
                    WholeMasks.append(part_folder + '/' + scan)
                if part == 'THIGH' and scan == 'MUSCLECOMP_mask.nii.gz':
                    ThighCompMasks.append(part_folder + '/' + scan)
                if part == 'CALF' and scan == 'MUSCLECOMP_mask.nii.gz':
                    CalfCompMasks.append(part_folder + '/' + scan)                       
                    
    return FatImgs, WholeMasks, ThighCompMasks, CalfCompMasks



def INTERSECTION(main_folder, app_folder, QUANTIFY_IMAT):
    
    FatImgs, WholeMasks, ThighCompMasks, CalfCompMasks = get_files_tmp(main_folder)
    
    # create a volumes matrix to save the MAT, Intra- and InterMAT volumes for each sample
    v_matrix = []
    v_matrix.append(['Fat image', 'V(MAT)', 'V(InterMAT)', 'V(IntraMAT)'])
    
    for idx in range(len(FatImgs)):
        '''
        ##################################################################################################################
        QUANTIFY MAT
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
        
        # find intersection by multiplying the arrays
        MAT = fat * wholemuscle
                
        # save MAT mask
        MAT_nifti = nib.Nifti1Image(MAT, affine=None, header=header, dtype=np.uint16)
        MAT_dir = WholeMasks[idx].replace('WHOLEMUSCLE_SAT', 'MAT')
        nib.save(MAT_nifti, MAT_dir)
        print(f'=> saved MAT mask: {MAT_dir}')
        
        # save MAT volume in volumes matrix
        new_row = [FatImgs[idx], (MAT.sum())*header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3]]

        if QUANTIFY_IMAT and 'THIGH' in FatImgs[idx] and len(ThighCompMasks)>0:
            '''
            ##################################################################################################################
            QUANTIFY Thigh Muscle Compartment IntraMAT
            '''
            # prepare muscle compartment mask
            mask, header = read_nifti_file(ThighCompMasks[idx//2])
            mask = mask.numpy()
                    
            # find intersection
            IntraMAT = fat * mask
                    
            # save IntraMAT mask
            IntraMAT_nifti = nib.Nifti1Image(IntraMAT, affine=None, header=header, dtype=np.uint16)
            IntraMAT_dir = ThighCompMasks[idx//2].replace('MUSCLECOMP', 'IntraMAT_COMP')
            nib.save(IntraMAT_nifti, IntraMAT_dir)
            print(f'=> saved IntraMAT mask: {IntraMAT_dir}')
            '''
            ##################################################################################################################
            QUANTIFY Thigh InterMAT
            '''               
            IntraMAT[IntraMAT > 0] = 1
            InterMAT = np.zeros_like(MAT)
            
            # find intersection
            MAT_map = 2*MAT - IntraMAT
            IntraMAT[MAT_map == 1] = 2
            IntraMAT[MAT_map < 0] = 0   # because there are edge regions which can be seen as IntraMAT but not as MAT
            InterMAT[MAT_map == 2] = 1
                    
            # save IMAT mask (IMAT = InterMAT + IntraMAT)
            IMAT = InterMAT + IntraMAT
            IMAT_nifti = nib.Nifti1Image(IMAT, affine=None, header=header, dtype=np.uint16)
            IMAT_dir = ThighCompMasks[idx//2].replace('MUSCLECOMP', 'Inter-&IntraMAT')
            nib.save(IMAT_nifti, IMAT_dir)
            print(f'=> saved Inter-&IntraMAT mask: {IMAT_dir}')
            
            # save Inter- & IntraMAT volumes in volumes matrix
            new_row.append((InterMAT.sum())*header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3])
            new_row.append((IntraMAT.sum()//2)*header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3])


        elif QUANTIFY_IMAT and 'CALF' in FatImgs[idx] and len(CalfCompMasks)>0:
            '''
            ##################################################################################################################
            QUANTIFY Calf Muscle Compartment IntraMAT
            '''
            # prepare muscle compartment mask
            mask, header = read_nifti_file(CalfCompMasks[idx//2])
            mask = mask.numpy()
                        
            # find intersection
            IntraMAT = fat * mask
            
            # save IntraMAT mask
            IntraMAT_nifti = nib.Nifti1Image(IntraMAT, affine=None, header=header, dtype=np.uint16)
            IntraMAT_dir = CalfCompMasks[idx//2].replace('MUSCLECOMP', 'IntraMAT_COMP')
            nib.save(IntraMAT_nifti, IntraMAT_dir)
            print(f'=> saved IntraMAT mask: {IntraMAT_dir}')
            '''
            ##################################################################################################################
            QUANTIFY Calf InterMAT
            '''
            IntraMAT[IntraMAT > 0] = 1
            InterMAT = np.zeros_like(MAT)

            # find intersection
            MAT_map = 2*MAT - IntraMAT
            IntraMAT[MAT_map == 1] = 2
            IntraMAT[MAT_map < 0] = 0   # because there are edge regions which can be seen as IntraMAT but not as MAT
            InterMAT[MAT_map == 2] = 1
                 
            # save IMAT mask (IMAT = InterMAT + IntraMAT)
            IMAT = InterMAT + IntraMAT
            IMAT_nifti = nib.Nifti1Image(IMAT, affine=None, header=header, dtype=np.uint16)
            IMAT_dir = CalfCompMasks[idx//2].replace('MUSCLECOMP', 'Inter-&IntraMAT')
            nib.save(IMAT_nifti, IMAT_dir)
            print(f'=> saved Inter-&IntraMAT mask: {IMAT_dir}')
            
            # save Inter- & IntraMAT volumes in volumes matrix
            new_row.append((InterMAT.sum())*header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3])
            new_row.append((IntraMAT.sum()//2)*header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3])
            
        else:
            new_row.extend([0, 0])        
        v_matrix.append(new_row)
      
    # save volumes matrix in an excel sheet
    file_path = f"{app_folder}/MAT_volumes.xlsx"
    v_matrix_df = pd.DataFrame(v_matrix[1:], columns=v_matrix[0])
    writer = pd.ExcelWriter(file_path, engine="xlsxwriter") 
    v_matrix_df.to_excel(writer, sheet_name="MAT volumes", index=False)
    writer.close()