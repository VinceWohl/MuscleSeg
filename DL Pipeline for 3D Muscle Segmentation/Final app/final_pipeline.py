# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:37:29 2023

@author: tp-vincentw

The pipeline involves five subsequent modules:
    - Model A: SAT and Whole Muscle segmentation in thigh and calf
    - Bridge: Necessary preparation for Model B and C by taking the output of Model A as reference
    - Model B: Thigh Muscle Compartments segmentation
    - Model C: Calf Muscle Compartments segmentation
    - Intersection: Quantification of MAT, IntraMAT and InterMAT
"""

import time
import torch
import INTERSECTION
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
    
    
    
def main():
        
    SOURCE_FOLDER = "/media/yeshe/Expansion/Work_during_PhD/Projects/Interns/Vinent/04 Python scripts/00 Pipeline development/AIPS Data 20th April 2022"
    APPLY_MODEL_A = False
    APPLY_MODEL_B = False
    APPLY_MODEL_C = False
    
    APP_FOLDER=""
    CHECKPOINT_A=""
    CHECKPOINT_B=""
    CHECKPOINT_C=""
    
    APPLY_PIPELINE(
    SOURCE_FOLDER,
    APP_FOLDER,
    APPLY_MODEL_A,
    CHECKPOINT_A,
    APPLY_MODEL_B,
    CHECKPOINT_B,
    APPLY_MODEL_C,
    CHECKPOINT_C,
    True,
    False)
        
        
if __name__ == "__main__":
    main()