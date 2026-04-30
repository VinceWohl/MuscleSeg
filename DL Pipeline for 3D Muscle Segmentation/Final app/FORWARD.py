# -*- coding: utf-8 -*-
"""
3D Muscle Segmentation Pipeline - Application Script

This is the only script you need to edit. Configure the paths and pipeline
stages below, then run:  python FORWARD.py

Required folder structure inside SOURCE_FOLDER:
    SOURCE_FOLDER/
        Subject_01/
            Calf/
                Water.nii.gz
                Fat.nii.gz
                In_phase.nii.gz
                Opp_phase.nii.gz
            Thigh/
                Water.nii.gz
                Fat.nii.gz
                In_phase.nii.gz
                Opp_phase.nii.gz
        Subject_02/
            ...

    - The pipeline auto-detects "Calf" and "Thigh" folders at any nesting
      depth, so intermediate levels (e.g. visits) are handled automatically.
    - Folder names must be exactly "Calf" and "Thigh" (case-insensitive).
    - Each folder must contain the four Dixon MRI scans as NIfTI files:
      Water.nii.gz, Fat.nii.gz, In_phase.nii.gz, Opp_phase.nii.gz

Pipeline stages (set True/False):
    Model A  -> Segments SAT and whole muscle using Water + Fat scans
    Bridge   -> Crops images around the whole muscle region (runs automatically when Model B or C is enabled)
    Model B  -> Segments 13 thigh muscle compartments using cropped Opp_phase + Water scans
    Model C  -> Segments 9 calf muscle compartments using cropped Opp_phase + Water scans
    MAT      -> Quantifies muscular adipose tissue using Fat image overlay on whole muscle mask
    IMAT     -> Quantifies inter- and intramuscular adipose tissue (requires Model B/C output)
"""

from final_pipeline import APPLY_PIPELINE

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Path to the folder containing your subject data (see folder structure above)
SOURCE_FOLDER = 'SOURCE_FOLDER'

# Path to the folder containing the pretrained model checkpoints (.pth files)
# Output Excel files with quantified volumes will also be saved here
APP_FOLDER = 'APP_FOLDER'

# ============================================================================
# PIPELINE STAGES - set to True to enable, False to skip
# ============================================================================

APPLY_PIPELINE(
    SOURCE_FOLDER=SOURCE_FOLDER,
    APP_FOLDER=APP_FOLDER,

    # Stage 1: SAT & Whole Muscle segmentation (uses Water + Fat scans)
    APPLY_MODEL_A=False,
    CHECKPOINT_A='lower-limb-muscle-and-sat-segmentation.pth',

    # Stage 2: Thigh muscle compartment segmentation (uses Opp_phase + Water, cropped by Bridge)
    # Output mask is restored to original image dimensions and can be overlaid on the uncropped scans
    APPLY_MODEL_B=False,
    CHECKPOINT_B='thigh-muscle-segmentation.pth',

    # Stage 3: Calf muscle compartment segmentation (uses Opp_phase + Water, cropped by Bridge)
    # Output mask is restored to original image dimensions and can be overlaid on the uncropped scans
    APPLY_MODEL_C=False,
    CHECKPOINT_C='calf-muscle-segmentation.pth',

    # Stage 4: MAT quantification (requires Model A output)
    QUANTIFY_MAT=True,

    # Stage 5: Inter- & IntraMAT quantification (requires Model B/C output)
    QUANTIFY_IMAT=True,
)
