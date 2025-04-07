# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:05:03 2023

@author: tp-vincentw

This script provides neccessary functions.

"""

import os
import torch
import numpy as np
import pandas as pd
import torchio as tio
import nibabel as nib
import torch.nn.functional as F

np.random.seed(0)


'''
######################################################################################################################################################
GATHERING IMAGE FILES
'''
def get_files_A(main_folder):       # for NIMMI data folder
    FirCImgs = []
    SecCImgs = []
    
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
                        if scan == 'roi_70p_thigh_W.nii.gz' or scan == 'roi_70p_calf_W.nii.gz':
                            FirCImgs.append(part_folder + '/' + scan)
                        if scan == 'roi_70p_thigh_F.nii.gz' or scan == 'roi_70p_calf_F.nii.gz':
                            SecCImgs.append(part_folder + '/' + scan)  
    imgs = [FirCImgs, SecCImgs]
    return imgs      


def get_files_B(main_folder):       # for NIMMI data folder
    FirCImgs = []
    SecCImgs = []

    subjects = sorted(os.listdir(main_folder))
    for subject in subjects:
        subject_folder = main_folder + '/' + subject
        visits = sorted(os.listdir(subject_folder))
        for visit in visits:
            visit_folder = subject_folder + '/' + visit
            parts = sorted(os.listdir(visit_folder))
            for part in parts:
                if part == 'THIGH':
                    part_folder = visit_folder + '/' + part
                    scans = sorted(os.listdir(part_folder))
                    for scan in scans:
                        if scan == 'roi_cropepd_70p_thigh_opp.nii.gz': FirCImgs.append(part_folder + '/' + scan)
                        if scan == 'roi_cropped_70p_thigh_W.nii.gz': SecCImgs.append(part_folder + '/' + scan) 
    imgs = [FirCImgs, SecCImgs]
    return imgs


def get_files_C(main_folder):       # for NIMMI data folder
    FirCImgs = []
    SecCImgs = []
    
    subjects = sorted(os.listdir(main_folder))
    for subject in subjects:
        subject_folder = main_folder + '/' + subject
        visits = sorted(os.listdir(subject_folder))
        for visit in visits:
            visit_folder = subject_folder + '/' + visit
            parts = sorted(os.listdir(visit_folder))
            for part in parts:
                if part == 'CALF':
                    part_folder = visit_folder + '/' + part
                    scans = sorted(os.listdir(part_folder))
                    for scan in scans:
                        if scan == 'roi_cropepd_70p_calf_opp.nii.gz': FirCImgs.append(part_folder + '/' + scan)
                        if scan == 'roi_cropped_70p_calf_W.nii.gz': SecCImgs.append(part_folder + '/' + scan) 
    imgs = [FirCImgs, SecCImgs]
    return imgs 


'''
######################################################################################################################################################
PREPROCESSING
'''
def read_nifti_file(path):
    nifti = nib.load(path)
    header = nifti.header.copy()
    array = nifti.get_fdata()
    tensor = torch.from_numpy(array)
    return tensor, header


class CropOrPadXY(tio.Transform):
    def __init__(self, app_shape):
        super().__init__()
        self.width = app_shape[0]
        self.height = app_shape[1]
        
    def apply_transform(self, subject):
        img = subject['image']
        img_array = img.numpy()
        
        # X_dim
        x_dim = img_array.shape[1]
        if x_dim <= self.width:
            # pad x dim
            x_pad = self.width - x_dim
            xa = x_pad // 2
            xb = x_pad - xa
            img_array = np.pad(img_array, ((0, 0), (xa, xb), (0, 0), (0, 0)), mode='constant')
        else:
            # crop x dim
            start_x = (x_dim - self.width) // 2
            end_x = start_x + self.width
            img_array = img_array[:, start_x:end_x, :, :]
              
        # Y_dim
        y_dim = img_array.shape[2]
        if y_dim <= self.height:
            # pad y dim
            y_pad = self.height - y_dim
            ya = y_pad // 2
            yb = y_pad - ya
            img_array = np.pad(img_array, ((0, 0), (0, 0), (ya, yb), (0, 0)), mode='constant')
        else:
            # crop y dim
            start_y = (y_dim - self.height) // 2
            end_y = start_y + self.height
            img_array = img_array[:, :, start_y:end_y, :]
        
        img = torch.from_numpy(img_array)
        subject['image'] = tio.Image(tensor=img, type=tio.INTENSITY)
        
        return subject


def get_transform_fn(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH):
    
    app_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH) # tio.CropOrPad expects dimensions (W, H, D)
    
    # transformation operator for image data
    transform = tio.Compose([
        CropOrPadXY(app_shape),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    return transform


'''
######################################################################################################################################################
MODEL APPLICATION
'''
def apply_model(model, img, patch_size):
    z_dim = img.shape[4]
    n = z_dim//patch_size
    
    if n == 0:      # no patch, padding
        pad = patch_size - z_dim
        za = pad//2
        zb = pad - za
        
        padded_img = F.pad(img, (za, zb), mode='constant', value=0)
        with torch.no_grad():
            probmap = model(padded_img)
        probmap = probmap[:, :, :, :, za:-zb]
            
    elif n == 1:    # two patches
        ov = (2*patch_size - z_dim)//2   # half overlap
        rest = (2*patch_size - z_dim)%2

        patch_1 = img[:, :, :, :, :patch_size]
        patch_2 = img[:, :, :, :, -patch_size:]
        with torch.no_grad():
            probmap_1 = model(patch_1)
            probmap_2 = model(patch_2)
        probmap = torch.cat((probmap_1[:, :, :, :, :-(ov+rest)], probmap_2[:, :, :, :, ov:]), dim=4)
    
    elif n == 2:    # three patches
        seed = patch_size//2
        rest = z_dim%(2*patch_size)
        
        patch_1 = img[:, :, :, :, :patch_size]
        patch_2 = img[:, :, :, :, seed+rest:seed+rest+patch_size]
        patch_3 = img[:, :, :, :, patch_size+rest:]
        with torch.no_grad():
            probmap_1 = model(patch_1)
            probmap_2 = model(patch_2)
            probmap_3 = model(patch_3)
        probmap = torch.cat((probmap_1[:, :, :, :, :seed+(2*rest)], probmap_2[:, :, :, :, rest:seed+(2*rest)], probmap_3[:, :, :, :, 2*rest:]), dim=4)
        
    return probmap


def retransform(pred, init_size):
    # X_dim
    if init_size[0] > pred.shape[0]: # pad
        x_pad = init_size[0] - pred.shape[0]
        xa = x_pad // 2
        xb = x_pad - xa
        pred = np.pad(pred, ((xa, xb), (0, 0), (0, 0)), mode='constant')
    elif init_size[0] < pred.shape[0]: # crop
        start_x = (pred.shape[0] - init_size[0]) // 2
        end_x = start_x + init_size[0]
        pred = pred[start_x:end_x, :, :]
        
    # Y_dim
    if init_size[1] > pred.shape[1]:
        y_pad = init_size[1] - pred.shape[1]
        ya = y_pad // 2
        yb = y_pad - ya
        pred = np.pad(pred, ((0, 0), (ya, yb), (0, 0)), mode='constant')
    else:
        start_y = (pred.shape[1] - init_size[1]) // 2
        end_y = start_y + init_size[1]
        pred = pred[:, start_y:end_y, :]
        
    return pred

def restore_orig_shape(pred, shape_diff):
    # shape_diff[min_x, fat_array.shape[0]-max_x, min_y, fat_array.shape[1]-max_y]
    # X_dim
    xa = shape_diff[0]
    xb = shape_diff[1]
    # Y_dim
    ya = shape_diff[2]
    yb = shape_diff[3]
    
    pred = np.pad(pred, ((xa, xb), (ya, yb), (0, 0)), mode='constant')
    return pred


def model_application(model, dic, app_ds, patch_size, app_folder, mID, shape_diffs=None, device='cuda'):
    model.eval()
    
    # create a classwise matrix to store the class dice scores and the class volumes for every test sample
    cw_matrix = []
    head_row = ['Image']
    for class_id, class_ in dic.items():
        head_row.append(f"V({class_['name']})/mm^3")
    cw_matrix.append(head_row)
    
    for idx in range(len(app_ds)):
        img = app_ds[idx]
        header_img = app_ds.get_headers(idx)
        
        img = img.to(device=device).unsqueeze(0)    # torch.tensor (B, N, H, W, D)
        
        # apply model
        probmap = apply_model(model, img, patch_size)
        
        # apply the argmax function on mask to get a 1-channel tensor mask
        probmap = probmap.squeeze(0)
        pred = torch.argmax(probmap, dim=0)
        pred_array = pred.cpu().numpy()
        
        # restransform to initial size
        pred_array = retransform(pred_array, app_ds.get_init_WnH(idx))

        # restore original image shape and manage filenames
        if mID == 'A':
            pred_dir = app_ds.FirCImgs[idx].replace(app_ds.FirCImgs[idx].split('/')[-1], 'WHOLEMUSCLE_SAT_mask.nii.gz')
            excel_dir = f'{app_folder}/WHOLEMUSCLE_SAT_volumes.xlsx'
        elif mID == 'B':
            pred_dir = app_ds.FirCImgs[idx].replace(app_ds.FirCImgs[idx].split('/')[-1], 'MUSCLECOMP_mask.nii.gz')
            excel_dir = f'{app_folder}/THIGH_MUSCLECOMP_volumes.xlsx'
            print(f"-> restore original shape according: {shape_diffs['WholeMuscle masks'][idx*2+1]}")
            pred_array = restore_orig_shape(pred_array, shape_diffs['buffer'][idx*2+1])
        elif mID == 'C':
            pred_dir = app_ds.FirCImgs[idx].replace(app_ds.FirCImgs[idx].split('/')[-1], 'MUSCLECOMP_mask.nii.gz')
            excel_dir = f'{app_folder}/CALF_MUSCLECOMP_volumes.xlsx'
            print(f"-> restore original shape with according: {shape_diffs['WholeMuscle masks'][idx*2]}")
            pred_array = restore_orig_shape(pred_array, shape_diffs['buffer'][idx*2])
        
        # determine segmented volumes and save them in classwise matrix
        new_row = [pred_dir]
        unique_intensities, intensity_counts = np.unique(pred_array, return_counts=True)
        for i in range(len(intensity_counts)):
            v = intensity_counts[i]*(header_img['pixdim'][1]*header_img['pixdim'][2]*header_img['pixdim'][3])
            new_row.append(v)

        # convert array to nifti and save it
        pred_nifti = nib.Nifti1Image(pred_array, affine=None, header=header_img, dtype=np.uint16)
        print(f'=> Save prediction: {pred_dir}')
        nib.save(pred_nifti, pred_dir)
        
        cw_matrix.append(new_row)
    
    # save cw matrix in an excel file
    cw_matrix_df = pd.DataFrame(cw_matrix[1:], columns=cw_matrix[0])
    writer = pd.ExcelWriter(excel_dir, engine="xlsxwriter")
    cw_matrix_df.to_excel(writer, sheet_name='Classwise volumes', index=False)
    writer.close()
    
    model.train()