# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:37:29 2023

@author: tp-vincentw

This scripts applies the end-to-end image processing pipeline involving
    - Model A: SAT and Whole Muscle segmentation
    - Bridge: Preparation step for Model B+C application by taking the output of Model A as input
    - Model B: Thigh Muscle Compartments segmentation
    - Model C: Calf Muscle Compartments segmentation
    - Intersection: Quantification of MAT, IntraMAT and InterMAT
"""

import torch
from dataset import MRI_Dataset
from model_A import UNET as MODEL_A
from model_B import UNET as MODEL_B
from model_C import UNET as MODEL_C
from bridge import BRIDGE
from intersection import INTERSECTION

from transform_fns import ( 
    get_transforms_A,
    # get_transforms_BnC,
    )


from utils import (
    get_files_A_app,
    get_files_B_app,
    get_files_C_app,
    model_application
    )
from loss_fns import DiceScore

'''
##################################################################################################################
Set the hyperparameters of the application session
'''
ID = 1
APPLY_MODEL_A = False
APPLY_BRIDGE = False
APPLY_MODEL_BnC = False
APPLY_INTERSECTION = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     # device which conducts the computation 
# free GPU memory
if DEVICE == 'cuda':
    torch.cuda.empty_cache()

DATA_PATH = '/media/yeshe/Expansion/Work_during_PhD/Projects/Interns/Vinent/04 Python scripts/00 Pipeline development/AIPS Data 20th April 2022'                     # folder with the original image data (structure: cohort/subjects/bodyparts/imgs)
APP_FOLDER = '_application'                                 # folder with the checkpoints of the trained models and as working folder
TEST_FN = DiceScore()                                       # define test function for evaluation

''' 
##################################################################################################################
Applicaiton parameters for Model A
'''
I_WIDTH_A = 368   # original 384
I_HEIGHT_A = 256  # original 288
I_DEPTH_A = 32    # original 45-68
CHECKPOINT_A = f'{APP_FOLDER}/cp_model_Av1_epoch_1063.pth'      # checkpoint file
DIC_A = {
       0: {'name': 'Clear Label',   'score': 0.0},
       1: {'name': 'SAT',           'score': 0.0},
       2: {'name': 'Whole Muscle',  'score': 0.0}
       }
NUM_CLASSES_A = len(DIC_A)

'''
##################################################################################################################
Applicaiton parameters for Model B
'''
I_WIDTH_B = 208   # original <384
I_HEIGHT_B = 208  # original <288
I_DEPTH_B = 48    # original 45-68
CHECKPOINT_B =  f'{APP_FOLDER}/cp_model_Bv1_epoch_xxxx.pth'     # checkpoint file
DIC_B = {
       0: {'name': 'Clear Label',                           'score': 0.0},
       1: {'name': 'Rectus Femoris',                        'score': 0.0},
       2: {'name': 'Vastus Lateralis',                      'score': 0.0},
       3: {'name': 'Vastus Intermedius',                    'score': 0.0},
       4: {'name': 'Vastus Medialis',                       'score': 0.0},
       5: {'name': 'Sartorius',                             'score': 0.0},
       6: {'name': 'Gracilis',                              'score': 0.0},
       7: {'name': 'Biceps Femoris',                        'score': 0.0},
       8: {'name': 'Semitendinosus',                        'score': 0.0},
       9: {'name': 'Semimembranosus',                       'score': 0.0},
       10: {'name': 'Adductor Brevis / Pectineus',          'score': 0.0},
       11: {'name': 'Adductor Longus',                      'score': 0.0},
       12: {'name': 'Adductor Magnus / Quadratus Femoris',  'score': 0.0},
       13: {'name': 'Gluteus Maximus',                      'score': 0.0}
       }
NUM_CLASSES_B = len(DIC_B)

'''
##################################################################################################################
Applicaiton parameters for Model B
'''
I_WIDTH_C = 160   # original <384
I_HEIGHT_C = 144  # original <288
I_DEPTH_C = 48    # original 45-68
CHECKPOINT_C = f'{APP_FOLDER}/cp_model_Cv1_epoch_xxxx.pth'      # checkpoint file
DIC_C = {
       0: {'name': 'Clear Label',                           'score': 0.0},
       1: {'name': 'Gastrocnemius Medialis',                'score': 0.0},
       2: {'name': 'Gastrocnemius Lateralis',               'score': 0.0},
       3: {'name': 'Soleus',                                'score': 0.0},
       4: {'name': 'Flexor Digitorum Longus',               'score': 0.0},
       5: {'name': 'Flexor Hallucis Longus',                'score': 0.0},
       6: {'name': 'Tibialis Posterior',                    'score': 0.0},
       7: {'name': 'Peroneus Longus and Brevis',            'score': 0.0},
       8: {'name': 'Tibialis Anterior',                     'score': 0.0},
       9: {'name': 'Extensor Hallucis / Digitorum Longus',  'score': 0.0},
       }
NUM_CLASSES_C = len(DIC_C)

print(f'PIPELINE APPLICATION SESSION: {ID}\n')
print('SETUP done\n---------------------------------------------------------------------------------------------------------------')



def main():

    if APPLY_MODEL_A:
        print('=> Load images for A')
        IMGS, MASKS = get_files_A_app(DATA_PATH)
        _, _, TEST_TF = get_transforms_A(I_WIDTH_A, I_HEIGHT_A, I_DEPTH_A)
        app_ds = MRI_Dataset(IMGS, MASKS, transform=TEST_TF, num_classes=NUM_CLASSES_A)
        print('=> Load Model A')
        model = MODEL_A(in_channels=2, out_channels=NUM_CLASSES_A).to(DEVICE)
        checkpoint = torch.load(CHECKPOINT_A, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('=> Apply Model A')
        model_application(model, DIC_A, app_ds, TEST_FN, I_DEPTH_A, APP_FOLDER, mID='A', device=DEVICE)
        print('MODEL A done\n------------------------------------------------------------------------------------')
    
    
        if APPLY_BRIDGE:
            print('=> Apply Bridge')
            BRIDGE(DATA_PATH, APP_FOLDER)
            print('BRIDGE done\n------------------------------------------------------------------------------------')
    
    
            if APPLY_MODEL_BnC:
                print('=> Load images for B')
                IMGS, MASKS = get_files_B_app(APP_FOLDER)
                _, _, TEST_TF = get_transforms_BnC(I_WIDTH_B, I_HEIGHT_B, I_DEPTH_B)
                app_ds = MRI_Dataset(IMGS, MASKS, transform=TEST_TF, num_classes=NUM_CLASSES_B)
                print('=> Load Model B')
                model = MODEL_B(in_channels=2, out_channels=NUM_CLASSES_B).to(DEVICE)
                checkpoint = torch.load(CHECKPOINT_B, map_location=torch.device(DEVICE))
                model.load_state_dict(checkpoint['model_state_dict'])
                print('=> Apply Model B')
                model_application(model, DIC_B, app_ds, TEST_FN, I_DEPTH_B, APP_FOLDER, mID='B', device=DEVICE)
                print('MODEL B done\n------------------------------------------------------------------------------------')
        
        
                print('=> Load images for C')
                IMGS, MASKS = get_files_C_app(APP_FOLDER)
                _, _, TEST_TF = get_transforms_BnC(I_WIDTH_C, I_HEIGHT_C, I_DEPTH_C)
                app_ds = MRI_Dataset(IMGS, MASKS, transform=TEST_TF, num_classes=NUM_CLASSES_C)
                print('=> Load Model C')
                model = MODEL_C(in_channels=2, out_channels=NUM_CLASSES_C).to(DEVICE)
                checkpoint = torch.load(CHECKPOINT_C, map_location=torch.device(DEVICE))
                model.load_state_dict(checkpoint['model_state_dict'])
                print('=> Apply Model C')
                model_application(model, DIC_C, app_ds, TEST_FN, I_DEPTH_C, APP_FOLDER, mID='C', device=DEVICE)
                print('MODEL C done\n------------------------------------------------------------------------------------')
        
        
    if APPLY_INTERSECTION:
        print('=> Apply Intersection')
        INTERSECTION(DATA_PATH, FIND_IMAT=APPLY_MODEL_BnC)
        print('INTERSECTION done\n------------------------------------------------------------------------------------')


    
if __name__ == "__main__":
    main()