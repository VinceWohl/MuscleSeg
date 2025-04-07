# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:14:46 2023

@author: tp-vincentw

This script applies an end-to-end image processing pipeline which takes following parameters as input:
    - SOURCE_FOLDER: contains all the image data in the structure COHORT/SUBJECT/VISIT/(CALF or THIGH)/(W, F, in, opp scans).nii.gz
        !! the sequence of datapoints needs to be characterized by successive alternating calf and thigh, starting with calf (e.g. CALF, THIGH, CALF, THIGH, ...) 
    - APP_FOLDER: contains the checkpoints of Model A, B, C and stores the generated excel files with all the quantified volumes
    - APPLY_MODEL_A: specifies whether Model A should be applied (segments SAT and Whole Muscle volume in thigh and calf MR images)
    - APPLY_MODEL_B: specifies whether Model B should be applied (segments thigh muscle compartments in respective MR images)
    - APPLY_MODEL_C: specifies whether Model C should be applied (segements calf muscle compartments in respective MR images)
    - QUANTIFY_MAT: specifies whether MAT should be quantified (only feasible if Model A segmented SAT and Whole Muscle)
    - QUANTIFY_IMAT: specifies whether Intra- and InterMAT should be quantified (only feasible if Model B and Model C segmented muscle compartments)
"""

from final_pipeline import APPLY_PIPELINE

print('PIPELINE APPLICATION SESSION\n')

APPLY_PIPELINE(SOURCE_FOLDER = 'AIPS',
               APP_FOLDER = 'APP_FOLDER',
               
               APPLY_MODEL_A = True,
               CHECKPOINT_A = 'cp_model_Av1_epoch_1725.pth',
               APPLY_MODEL_B = False,
               CHECKPOINT_B = 'cp_model_Bv5_epoch_3299.pth',
               APPLY_MODEL_C = False,
               CHECKPOINT_C = 'cp_model_Cv3_epoch_4494.pth',
               QUANTIFY_MAT = True,
               QUANTIFY_IMAT = False
               )