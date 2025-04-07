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
import nibabel as nib
import torch.nn.functional as F
from dataset import MRI_Dataset
from torch.utils.data import DataLoader


######################################################################################################################################################
def get_files_A(main_folder):
    '''
    gets the images and masks for training of model A
    '''
    
    FirCImgs = []
    SecCImgs = []
    masks = []
    
    subjects = os.listdir(main_folder)  # use os.listdir to get a list of all the files and folders in a main folder
    subjects = sorted(subjects, key=lambda x: int(x))
    
    for subject in subjects:
        subject_path = main_folder + '/' + subject
                
        parts = os.listdir(subject_path)
        for part in parts:
            part_path = subject_path + '/' + part
            
            FirCImgs.append(part_path + '/' + 'roi_70p_Water.nii.gz')
            SecCImgs.append(part_path + '/' + 'roi_70p_Fat.nii.gz')
                
            if part == 'CALF':
                masks.append(part_path + '/' + subject + '_CALF_WHOLEMUSCLE_SAT_mask.nii.gz')
            if part == 'THIGH':
                masks.append(part_path + '/' + subject + '_THIGH_WHOLEMUSCLE_SAT_mask.nii.gz')
    
    imgs = [FirCImgs, SecCImgs]
    return imgs, masks


######################################################################################################################################################
def get_files_A_app(main_folder):
    '''
    gets the images and masks for application of model A
    '''
    FirCImgs = []
    SecCImgs = []
    masks = []
    
    subjects = os.listdir(main_folder)  # use os.listdir to get a list of all the files and folders in a main folder
    subjects = sorted(subjects, key=lambda x: int(x))
    
    for subject in subjects:
        subject_path = main_folder + '/' + subject
                
        parts = os.listdir(subject_path)
        for part in parts:
            part_path = subject_path + '/' + part
            
            FirCImgs.append(part_path + '/' + 'roi_70p_Water.nii.gz')
            SecCImgs.append(part_path + '/' + 'roi_70p_Fat.nii.gz')
                
            if part == 'CALF':
                masks.append(part_path + '/' + subject + '_CALF_WHOLEMUSCLE_SAT_mask.nii.gz')
            if part == 'THIGH':
                masks.append(part_path + '/' + subject + '_THIGH_WHOLEMUSCLE_SAT_mask.nii.gz')

    imgs = [FirCImgs, SecCImgs]  
    return imgs, masks
        

######################################################################################################################################################
def get_files_B(main_folder):
    '''
    gets the images and masks for training of model B
    '''
    FirCImgs = []
    SecCImgs = []
    masks = []
        
    subjects = os.listdir(main_folder)  # use os.listdir to get a list of all the files and folders in a main folder
    subjects = sorted(subjects, key=lambda x: int(x))
    
    for subject in subjects:
        subject_path = main_folder + '/' + subject
                
        parts = os.listdir(subject_path)
        for part in parts:
            part_path = subject_path + '/' + part
            
            if part == 'THIGH':
                FirCImgs.append(part_path + '/' + subject + '_THIGH_cropped_roi_70p_Opp_Phase.nii.gz')
                SecCImgs.append(part_path + '/' + subject + '_THIGH_cropped_roi_70p_Water.nii.gz')
                masks.append(part_path + '/' + subject + '_THIGH_MUSCLE_COMP_cropped_mask.nii.gz')
    
    imgs = [FirCImgs, SecCImgs]  
    return imgs, masks


######################################################################################################################################################
def get_files_B_app(app_folder):
    '''
    gets the images and masks for application of model B
    '''
    FirCImgs = []
    SecCImgs = []
    masks = []
    
    files = os.listdir(app_folder)
    
    for file in files:
        if '_THIGH_cropped_roi_70p_Opp_Phase.nii.gz' in file:
            FirCImgs.append(f'{app_folder}/{file}')
            
        if '_THIGH_cropped_roi_70p_Water.nii.gz' in file:
            SecCImgs.append(f'{app_folder}/{file}')

        if '_THIGH_MUSCLE_COMP_cropped_mask.nii.gz' in file:
            masks.append(f'{app_folder}/{file}')
    
    FirCImgs = sorted(FirCImgs)
    SecCImgs = sorted(SecCImgs)
    masks = sorted(masks)
    
    imgs = [FirCImgs, SecCImgs]  
    return imgs, masks


######################################################################################################################################################
def get_files_C(main_folder):
    '''
    gets the images and masks for training of model C
    '''
    FirCImgs = []
    SecCImgs = []
    masks = []
        
    subjects = os.listdir(main_folder)  # use os.listdir to get a list of all the files and folders in a main folder
    subjects = sorted(subjects, key=lambda x: int(x))
    
    for subject in subjects:
        subject_path = main_folder + '/' + subject
                
        parts = os.listdir(subject_path)
        for part in parts:
            part_path = subject_path + '/' + part
            
            if part == 'CALF':
                FirCImgs.append(part_path + '/' + subject + '_CALF_cropped_roi_70p_Opp_Phase.nii.gz')
                SecCImgs.append(part_path + '/' + subject + '_CALF_cropped_roi_70p_Water.nii.gz')
                masks.append(part_path + '/' + subject + '_CALF_MUSCLE_COMP_cropped_mask.nii.gz')
        
    imgs = [FirCImgs, SecCImgs]  
    return imgs, masks


######################################################################################################################################################
def get_files_C_app(app_folder):
    '''
    gets the images and masks for application of model C
    '''
    FirCImgs = []
    SecCImgs = []
    masks = []

    files = os.listdir(app_folder)
    
    for file in files:
        if '_CALF_cropped_roi_70p_Opp_Phase.nii.gz' in file:
            FirCImgs.append(f'{app_folder}/{file}')
            
        if '_CALF_cropped_roi_70p_Water.nii.gz' in file:
            SecCImgs.append(f'{app_folder}/{file}')

        if '_CALF_MUSCLE_COMP_cropped_mask.nii.gz' in file:
            masks.append(f'{app_folder}/{file}')
    
    FirCImgs = sorted(FirCImgs)
    SecCImgs = sorted(SecCImgs)
    masks = sorted(masks)
    
    imgs = [FirCImgs, SecCImgs]  
    return imgs, masks


######################################################################################################################################################
def splitup(imgs, masks, folder, split):
    
    FirCImgs = imgs[0][:]
    SecCImgs = imgs[1][:]
    n = len(masks)//5
    
    ### Splitting up the five folds (1-5) for the 5-fold cross validation
    f1_i = [FirCImgs[:n], SecCImgs[:n]]
    f1_m = masks[:n]

    f2_i = [FirCImgs[n:n*2], SecCImgs[n:n*2]]
    f2_m = masks[n:n*2]
    
    f3_i = [FirCImgs[n*2:n*3], SecCImgs[n*2:n*3]]
    f3_m = masks[n*2:n*3]
    
    f4_i = [FirCImgs[n*3:n*4], SecCImgs[n*3:n*4]]
    f4_m = masks[n*3:n*4]
    
    f5_i = [FirCImgs[n*4:n*5], SecCImgs[n*4:n*5]]
    f5_m = masks[n*4:n*5]
    
    if split == 1:
        train_imgs = [f3_i[0]+f4_i[0]+f5_i[0], f3_i[1]+f4_i[1]+f5_i[1]]
        train_masks = f3_m + f4_m + f5_m
        val_imgs = f2_i
        val_masks = f2_m
        test_imgs = f1_i
        test_masks = f1_m
    elif split == 2:
        train_imgs = [f1_i[0]+f4_i[0]+f5_i[0], f1_i[1]+f4_i[1]+f5_i[1]]
        train_masks = f1_m + f4_m + f5_m
        val_imgs = f3_i
        val_masks = f3_m
        test_imgs = f2_i
        test_masks = f2_m
    elif split == 3:
        train_imgs = [f1_i[0]+f2_i[0]+f5_i[0], f1_i[1]+f2_i[1]+f5_i[1]]
        train_masks = f1_m + f2_m + f5_m
        val_imgs = f4_i
        val_masks = f4_m
        test_imgs = f3_i
        test_masks = f3_m
    elif split == 4:
        train_imgs = [f1_i[0]+f2_i[0]+f3_i[0], f1_i[1]+f2_i[1]+f3_i[1]]
        train_masks = f1_m + f2_m + f3_m
        val_imgs = f5_i
        val_masks = f5_m
        test_imgs = f4_i
        test_masks = f4_m
    elif split == 5:
        train_imgs = [f2_i[0]+f3_i[0]+f4_i[0], f2_i[1]+f3_i[1]+f4_i[1]]
        train_masks = f2_m + f3_m + f4_m
        val_imgs = f1_i
        val_masks = f1_m
        test_imgs = f5_i
        test_masks = f5_m
    else:
        print("Split index out of range!")
    
    
    train_df = pd.DataFrame({"Train_FirstChannel": train_imgs[0][:], "Train_SecondChannel": train_imgs[1][:], "Train_Masks": train_masks})
    val_df = pd.DataFrame({"Val_FirstChannel": val_imgs[0][:], "Val_SecondChannel": val_imgs[1][:], "Val_Masks": val_masks})
    test_df = pd.DataFrame({"Test_FirstChannel": test_imgs[0][:], "Test_SecondChannel": test_imgs[1][:], "Test_Masks": test_masks})

    file_path = f"{folder}/split_{split}.xlsx"
    writer = pd.ExcelWriter(file_path, engine="xlsxwriter")

    print(f"=> Saving files directions of split {split}")
    train_df.to_excel(writer, sheet_name="Training", index=False)
    val_df.to_excel(writer, sheet_name="Validation", index=False)
    test_df.to_excel(writer, sheet_name="Testing", index=False)
    writer.close()
    
    return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks


######################################################################################################################################################
def read_nifti_file(path):
    nifti = nib.load(path)
    header = nifti.header.copy()
    array = nifti.get_fdata()
    tensor = torch.from_numpy(array)
    return tensor, header


######################################################################################################################################################
def get_loaders(
        train_imgdir,
        train_maskdir,
        val_imgdir,
        val_maskdir,
        
        num_classes,
        
        train_transform,
        val_transform,
        
        batch_size,
        
        num_workers=4,
        pin_memory=True,
    ):
        
    # Training data
    train_ds = MRI_Dataset(
        image_dir=train_imgdir,
        mask_dir=train_maskdir,
        num_classes=num_classes,
        transform=train_transform
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        )
        
    # Validation data
    val_ds = MRI_Dataset(
        image_dir=val_imgdir,
        mask_dir=val_maskdir,
        num_classes=num_classes,
        transform=val_transform,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        )
    
    return train_loader, val_loader


######################################################################################################################################################
def validate(model, loader, loss_fn, device="cuda"):
    model.eval()    
    v_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device=device)       # torch.tensor (B, C, W, H, D)
        target = target.to(device=device)   # torch.tensor (B, C, W, H, D)
        with torch.no_grad():
            probmap = model(data)
        loss = loss_fn(probmap, target)
        v_loss += loss.item()
    
    model.train()
    return v_loss/len(loader)


######################################################################################################################################################
def save_scores(train_scores, val_scores, best_val, folder):
    
    train_val_df = pd.DataFrame({"Train_scores": train_scores, "Val_scores": val_scores})
    best_val_df = pd.DataFrame({"Best_val_score": [best_val[0]], "Epoch": [best_val[1]]})

    file_path = f"{folder}/train_and_val_scores.xlsx"
    writer = pd.ExcelWriter(file_path, engine="xlsxwriter")

    print("=> Saving training and validation scores")
    train_val_df.to_excel(writer, sheet_name="Train & Val scores", index=False)
    best_val_df.to_excel(writer, sheet_name="Best Validation", index=False)
    writer.close()

    
######################################################################################################################################################
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


######################################################################################################################################################
def model_application(model, dic, app_ds, test_fn, patch_size, folder, mID, device='cuda'):
    model.eval()
    test_score = 0.0
    
    # create a classwise matrix to store the class dice scores and the class volumes for every test sample
    cw_matrix = []
    head_row = ['Ground Truth']
    for class_id, class_ in dic.items():
        head_row.append(f"T({class_['name']})")
    for class_id, class_ in dic.items():
        head_row.append(f"V({class_['name']})/mm^3")
    cw_matrix.append(head_row)
    
    for idx in range(len(app_ds)):
        img, target = app_ds[idx]
        header_img, header_mask = app_ds.get_headers(idx)
        
        img = img.to(device=device).unsqueeze(0)        # torch.tensor (B, N, H, W, D)
        target = target.to(device=device).unsqueeze(0)  # torch.tensor (B, N, H, W, D)
        
        '''
        ##################################################################################################################
        PRED: apply the model on the reference image and save the prediction mask
        '''
        probmap = apply_model(model, img, patch_size)
        
        # determine test score
        dic, t_score = test_fn(dic, probmap, target)
        test_score += t_score.item()
        
        # apply the argmax function on mask to get a 1-channel tensor mask
        probmap = probmap.squeeze(0)
        pred = torch.argmax(probmap, dim=0)
        pred_array = pred.cpu().numpy()
        
        # save scores and volumes in classwise matrix
        new_row = [app_ds.masks[idx].replace('mask', 'pred')]
        for class_id, class_ in dic.items():
            new_row.append(class_['score'].item())
        unique_intensities, intensity_counts = np.unique(pred_array, return_counts=True)
        if len(intensity_counts) == len(dic):
            for i in range(len(intensity_counts)):
                v = intensity_counts[i]*(header_img['pixdim'][1]*header_img['pixdim'][2]*header_img['pixdim'][3])
                new_row.append(v)
        else:
            v = [0] * len(dic)
            new_row.extend(v)
        cw_matrix.append(new_row)
        
        '''
        ##################################################################################################################
        TARGET
        '''
        # apply the argmax function on mask to get a 1-channel tensor mask
        target = target.squeeze(0)
        target = torch.argmax(target, dim=0)
        target_array = target.cpu().numpy()

        '''
        ##################################################################################################################
        IMAGE
        '''
        # only the first channel image will be saved
        img = img[0, 0, :, :, :]
        img_array = img.cpu().numpy()
          
        
        # convert arrays to nifti and save them
        pred_nifti = nib.Nifti1Image(pred_array, affine=None, header=header_mask, dtype=np.uint16)
        target_nifti = nib.Nifti1Image(target_array, affine=None, header=header_mask, dtype=np.uint16)
        img_nifti = nib.Nifti1Image(img_array, affine=None, header=header_img, dtype=np.uint16)
        
        # manage file names
        pred_name = app_ds.masks[idx].split('/')[-1].replace('mask', 'pred')
        target_name = app_ds.masks[idx].split('/')[-1]
        sub_id = target_name.split('_')[0]
        part = target_name.split('_')[1]
        img_name = f'{sub_id}_{part}_first_channel_img_used_for_{mID}.nii.gz'
        
        print(f'=> Save prediction: {pred_name}')
        nib.save(pred_nifti, f"{folder}/{pred_name}")
        nib.save(target_nifti, f"{folder}/{target_name}")
        nib.save(img_nifti, f"{folder}/{img_name}")
        
     # determine the overall average test score
    test_score = test_score/len(app_ds)
    print(f"Model {mID} overall average test score: {test_score}")
    
    # save cw matrix and test score as dataframe and in an excel file
    test_df = pd.DataFrame({"Test_score": [test_score]})
    cw_matrix_df = pd.DataFrame(cw_matrix[1:], columns=cw_matrix[0])
    file_path = f"{folder}/Model_{mID}_results.xlsx"
    writer = pd.ExcelWriter(file_path, engine="xlsxwriter")
    print("=> Saving testing scores and volumes")
    test_df.to_excel(writer, sheet_name="Test score", index=False)
    cw_matrix_df.to_excel(writer, sheet_name="Classwise scores and volumes", index=False)
    writer.close()
    
    model.train()