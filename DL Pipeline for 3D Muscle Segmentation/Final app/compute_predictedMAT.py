# -*- coding: utf-8 -*-

from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import torch
import sys
import os 
# import INTERSECTION

from final_utils import(
    get_files_A,
    get_files_B,
    get_files_C,
    get_transform_fn,
    model_application,
    )


def APPLY_PIPELINE(SOURCE_FOLDER,
                   APP_FOLDER,
                   APPLY_MODEL_A,
                   CHECKPOINT_A,
                   APPLY_MODEL_B,
                   CHECKPOINT_B,
                   APPLY_MODEL_C,
                   CHECKPOINT_C,
                   QUANTIFY_MAT,
                   QUANTIFY_IMAT
                   ):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     # device which conducts the computation 
    # free GPU memory
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    ''' 
    ##################################################################################################################
    Parameters for Model A
    '''
    I_WIDTH_A = 368
    I_HEIGHT_A = 256
    I_DEPTH_A = 32
    DIC_A = {
           0: {'name': 'Clear Label'},
           1: {'name': 'SAT'},
           2: {'name': 'Whole Muscle'}
           }
    NUM_CLASSES_A = len(DIC_A)
    
    '''
    ##################################################################################################################
    Parameters for Model B
    '''
    I_WIDTH_B = 208
    I_HEIGHT_B = 208
    I_DEPTH_B = 48
    DIC_B = {
           0: {'name': 'Clear Label'},
           1: {'name': 'Rectus Femoris'},
           2: {'name': 'Vastus Lateralis'},
           3: {'name': 'Vastus Intermedius'},
           4: {'name': 'Vastus Medialis'},
           5: {'name': 'Sartorius'},
           6: {'name': 'Gracilis'},
           7: {'name': 'Biceps Femoris'},
           8: {'name': 'Semitendinosus'},
           9: {'name': 'Semimembranosus'},
           10: {'name': 'Adductor Brevis / Pectineus'},
           11: {'name': 'Adductor Longus'},
           12: {'name': 'Adductor Magnus / Quadratus Femoris'},
           13: {'name': 'Gluteus Maximus'}
           }
    NUM_CLASSES_B = len(DIC_B)
    
    '''
    ##################################################################################################################
    Parameters for Model C
    '''
    I_WIDTH_C = 160
    I_HEIGHT_C = 144
    I_DEPTH_C = 48
    DIC_C = {
           0: {'name': 'Clear Label'},
           1: {'name': 'Gastrocnemius Medialis'},
           2: {'name': 'Gastrocnemius Lateralis'},
           3: {'name': 'Soleus'},
           4: {'name': 'Flexor Digitorum Longus'},
           5: {'name': 'Flexor Hallucis Longus'},
           6: {'name': 'Tibialis Posterior'},
           7: {'name': 'Peroneus Longus and Brevis'},
           8: {'name': 'Tibialis Anterior'},
           9: {'name': 'Extensor Hallucis / Digitorum Longus'}
           }
    NUM_CLASSES_C = len(DIC_C)
    
    print(f'APPLY_MODEL_A: {APPLY_MODEL_A}\nAPPLY_MODEL_B: {APPLY_MODEL_B}\nAPPLY_MODEL_C: {APPLY_MODEL_C}\nQUANTIFY_MAT: {QUANTIFY_MAT}\nQUANTIFY_IMAT: {QUANTIFY_IMAT}')
    print('SETUP done\n---------------------------------------------------------------------------------------------------------------')
    start_time = time.time()


    if APPLY_MODEL_A:
        print('=> Prepare dataset for A')
        IMGS = get_files_A(SOURCE_FOLDER)
        TF = get_transform_fn(I_WIDTH_A, I_HEIGHT_A, I_DEPTH_A)
        app_ds = MRI_Dataset(IMGS, transform=TF)
        print('=> Load Model A')
        model = MODEL_A(in_channels=2, out_channels=NUM_CLASSES_A).to(DEVICE)
        checkpoint = torch.load(f'{APP_FOLDER}/{CHECKPOINT_A}', map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('=> Apply Model A')
        model_application(model, DIC_A, app_ds, I_DEPTH_A, APP_FOLDER, mID='A', device=DEVICE)
        print('MODEL A done\n------------------------------------------------------------------------------------')

    if APPLY_MODEL_B or APPLY_MODEL_C:
        print('=> Apply Bridge')
        SHAPE_DIFFS = BRIDGE(SOURCE_FOLDER)
        print('BRIDGE done\n------------------------------------------------------------------------------------')
        
    if APPLY_MODEL_B:
        print('=> Prepare dataset for B')
        IMGS = get_files_B(SOURCE_FOLDER)
        TF = get_transform_fn(I_WIDTH_B, I_HEIGHT_B, I_DEPTH_B)
        app_ds = MRI_Dataset(IMGS, transform=TF)
        print('=> Load Model B')
        model = MODEL_B(in_channels=2, out_channels=NUM_CLASSES_B).to(DEVICE)
        checkpoint = torch.load(f'{APP_FOLDER}/{CHECKPOINT_B}', map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('=> Apply Model B')
        model_application(model, DIC_B, app_ds, I_DEPTH_B, APP_FOLDER, mID='B', shape_diffs=SHAPE_DIFFS, device=DEVICE)
        print('MODEL B done\n------------------------------------------------------------------------------------')
            
    if APPLY_MODEL_C:
        print('=> Prepaere dataset for C')
        IMGS = get_files_C(SOURCE_FOLDER)
        TF = get_transform_fn(I_WIDTH_C, I_HEIGHT_C, I_DEPTH_C)
        app_ds = MRI_Dataset(IMGS, transform=TF)
        print('=> Load Model C')
        model = MODEL_C(in_channels=2, out_channels=NUM_CLASSES_C).to(DEVICE)
        checkpoint = torch.load(f'{APP_FOLDER}/{CHECKPOINT_C}', map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('=> Apply Model C')
        model_application(model, DIC_C, app_ds, I_DEPTH_C, APP_FOLDER, mID='C', shape_diffs=SHAPE_DIFFS, device=DEVICE)
        print('MODEL C done\n------------------------------------------------------------------------------------') 
        
    if QUANTIFY_MAT:
        print('=> Apply Intersection')
        INTERSECTION(SOURCE_FOLDER, APP_FOLDER, QUANTIFY_IMAT)
        print('INTERSECTION done\n------------------------------------------------------------------------------------')
   
    
    # display the running time
    end_time = time.time()
    running_time_minutes = (end_time - start_time)//60
    hours = int(running_time_minutes // 60)
    minutes = int(running_time_minutes % 60)
    print('COMPLETE\n---------------------------------------------------------------------------------------------------------------')
    print(f'Runtime for application: {hours}:{minutes}\n---------------------------------------------------------------------------------------------------------------')
    
    
from pathlib import Path

def compute_total_MAT(fat_path, whole_muscle_mask_path, limb):
    
    # ---------------------------------------------------------------------------
    # Load data 
    # ---------------------------------------------------------------------------
    fat_nii = nib.load(fat_path)
    pixel_dim = fat_nii.header['pixdim'][1:4]
    fat_img = fat_nii.get_fdata()
    # Normalize values in fat_img to the range [0, 1]
    fat_img = (fat_img - np.min(fat_img)) / (np.max(fat_img) - np.min(fat_img))
    # thresholding
    fat_img = np.where(fat_img < 0.1, 0, 1)
    
    
    whole_muscle_mask = load_nifi_volume(whole_muscle_mask_path)
    muscle_comp_mask = load_nifi_volume(muscle_comp_mask_path)
    
    # ---------------------------------------------------------------------------
    # Compute Total MAT
    # ---------------------------------------------------------------------------
    # Create Muscle only mask
    whole_muscle_mask = np.where(whole_muscle_mask == 2, 1, 0)
    # Multiply fat_img with mc_mask
    total_mat_img = fat_img * whole_muscle_mask
    # compute volume     
    total_mat_volume = np.sum(total_mat_img) * np.prod(pixel_dim)
    
    if save_mask:
        nib.save(nib.Nifti1Image(total_mat_img, affine=fat_nii.affine),
                 os.path.join(os.path.dirname(fat_path), 'MAT_mask.nii.gz'))
    
    # ---------------------------------------------------------------------------
    # Compute Intra-MAT
    # ---------------------------------------------------------------------------
    
    # find intersection
    total_intra_mat_img = np.where(muscle_comp_mask != 0, 1, 0)
    total_intra_mat_img[total_mat_img==0] = 0 # to make sure comparmtent masks are within total muscle mask  
    inter_mat_img = total_mat_img - total_intra_mat_img
    
    # compute volume 
    inter_mat_volume = np.sum(inter_mat_img) * np.prod(pixel_dim)
    
    # compute muscle specific Intra-MAT
    intra_mat_volumes = {}
    muscles_dic = thigh_dict if limb == "THIGH" else calf_dict
    
    for label_key, muscle_name in muscles_dic.items():
        tmp_muscle_mask = np.where(muscle_comp_mask == label_key,1, 0)
        tmp_intra_mat_img = fat_img * tmp_muscle_mask
        intra_mat_volumes[muscle_name] = np.sum(tmp_intra_mat_img) * np.prod(pixel_dim)
    
    return total_mat_volume, inter_mat_volume, intra_mat_volumes

def compute_MAT(data_path, dst_path):
    
    # Initialize empty lists to store the results for the current subject
    calf_results_list = []
    thigh_results_list = []
    
    for subject in tqdm(os.listdir(data_path)):
        subject_path = data_path/subject    
        
        for limb in ["CALF", "THIGH"]:
            subject_limb_path = subject_path/limb
             # compute whole muscle MAT and Muscle specific IntraMAT
            total_mat_volume, inter_mat_volume, intra_mat_volumes = compute_MAT(subject_limb_path/"roi_70p_Fat.nii.gz",
                                     subject_limb_path/(subject + "_" + limb + "_WHOLEMUSCLE_SAT_mask.nii.gz"), 
                                     subject_limb_path/ (subject + "_" + limb + "_MUSCLE_COMP_mask.nii.gz"),
                                     limb,
                                     False)
            
            # Create a dictionary for the current subject and limb
            subject_df = pd.DataFrame({
                    'subjectID': subject,
                    'Total MAT volume': [total_mat_volume],
                    'Inter MAT volume': [inter_mat_volume],
                    **intra_mat_volumes  # Unpack intra_mat_volumes dictionary
                })
             
            # Append the dictionary to the results list for the current limb
            if limb == "CALF":
                calf_results_list.append(subject_df)
            elif limb == "THIGH":
                thigh_results_list.append(subject_df)           
    # --------------------------------------------------
    # Save results 
    # Concatenate all DataFrames in calf_results_list into a single DataFrame and save
    pd.concat(calf_results_list, axis=0, ignore_index=True).to_excel(dst_path/"calf_mat_gt_volumes.xlsx")
    pd.concat(thigh_results_list, axis=0, ignore_index=True).to_excel(dst_path/"thigh_mat_gt_volumes.xlsx")      
    
def main():
        
    data_path = Path(r"/media/yeshe/Expansion/Work_during_PhD/Projects/Interns/Vinent/04 Python scripts/00 Pipeline development/AIPS Data 20th April 2022")
    compute_MAT(data_path)
    

        
        
if __name__ == "__main__":
    main()