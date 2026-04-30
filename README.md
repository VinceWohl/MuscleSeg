# MuscleSeg - 3D Lower-Limb Muscle Segmentation Pipeline

A deep learning pipeline for automated segmentation and quantification of lower-limb muscles from Dixon MRI scans. The pipeline segments subcutaneous adipose tissue (SAT), whole muscle, and individual muscle compartments in thigh and calf images, and quantifies muscular adipose tissue (MAT, IntraMAT, InterMAT).

## Links

- **Dataset**: https://doi.org/10.6084/m9.figshare.31042489
- **Paper**: https://lnkd.in/db_FZct4
- **Pretrained models**: https://huggingface.co/yeshekway/lower-limb-muscle-segmentation

## Repository Structure

```
MuscleSeg-main/
    DL Pipeline for 3D Muscle Segmentation/
        Pipeline development/     # Model training and development code
        Final app/                # Inference pipeline (see below)
```

## Pipeline Overview

The inference pipeline (`Final app/`) consists of five stages:

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| **Model A** | WholeMuscleNet | Water + Fat | SAT & whole muscle mask (3 classes) |
| **Bridge** | *(auto)* | Model A output | Cropped images around muscle region |
| **Model B** | ThighMuscleNet | Cropped Opp_phase + Water | 13 thigh muscle compartment mask |
| **Model C** | CalfMuscleNet | Cropped Opp_phase + Water | 9 calf muscle compartment mask |
| **Intersection** | *(computation)* | Fat image + masks | MAT, IntraMAT, InterMAT volumes |

All three models are 3D U-Nets with attention gates, trained on patch sizes of 32 (Model A) or 48 (Models B and C) slices.

### Segmented Muscle Compartments

**Thigh (Model B, 13 classes):** Rectus Femoris, Vastus Lateralis, Vastus Intermedius, Vastus Medialis, Sartorius, Gracilis, Biceps Femoris, Semitendinosus, Semimembranosus, Adductor Brevis/Pectineus, Adductor Longus, Adductor Magnus/Quadratus Femoris, Gluteus Maximus

**Calf (Model C, 9 classes):** Gastrocnemius Medialis, Gastrocnemius Lateralis, Soleus, Flexor Digitorum Longus, Flexor Hallucis Longus, Tibialis Posterior, Peroneus Longus and Brevis, Tibialis Anterior, Extensor Hallucis/Digitorum Longus

## Quick Start

### 1. Install dependencies

```bash
cd "DL Pipeline for 3D Muscle Segmentation/Final app"
pip install -r requirements.txt
```

> **Note on PyTorch:** The pipeline requires `torch>=2.6.0,<2.10.0`. Versions 2.10.0+ have a known issue with Conv3d (groups=2) on CPU that can cause crashes. Install the CPU-only build if you don't have a CUDA GPU:
> ```bash
> pip install "torch>=2.6.0,<2.10.0" --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. Download pretrained models

Download the three checkpoint files from [HuggingFace](https://huggingface.co/yeshekway/lower-limb-muscle-segmentation) and place them in the `APP_FOLDER/` directory:

- `lower-limb-muscle-and-sat-segmentation.pth` (Model A)
- `thigh-muscle-segmentation.pth` (Model B)
- `calf-muscle-segmentation.pth` (Model C)

### 3. Prepare your data

Organize your MRI scans into the following folder structure:

```
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
        Calf/
            ...
        Thigh/
            ...
```

- The pipeline auto-detects `Calf` and `Thigh` folders at any nesting depth, so intermediate levels (e.g. cohort, visit) are supported automatically.
- Folder names must be `Calf` and `Thigh` (case-insensitive).
- Each folder must contain the four Dixon MRI sequences as NIfTI files: `Water.nii.gz`, `Fat.nii.gz`, `In_phase.nii.gz`, `Opp_phase.nii.gz`.
- Model A uses `Water` + `Fat`, Models B and C use `Opp_phase` + `Water` (cropped automatically by the Bridge stage).

### 4. Configure and run

Edit `FORWARD.py` -- it is the only file you need to modify:

```python
# Set the paths to your data and checkpoint folders
SOURCE_FOLDER = 'path/to/your/data'
APP_FOLDER = 'path/to/your/checkpoints'

# Enable the pipeline stages you need
APPLY_MODEL_A = True    # SAT & whole muscle segmentation
APPLY_MODEL_B = True    # Thigh compartment segmentation
APPLY_MODEL_C = True    # Calf compartment segmentation
QUANTIFY_MAT  = True    # MAT quantification
QUANTIFY_IMAT = True    # Inter- & IntraMAT quantification
```

Then run:

```bash
python FORWARD.py
```

## Output

### Segmentation masks (saved in each subject's Calf/Thigh folder)

| File | Source |
|------|--------|
| `WHOLEMUSCLE_SAT_mask.nii.gz` | Model A |
| `MUSCLECOMP_mask.nii.gz` | Model B / Model C |
| `MAT_mask.nii.gz` | Intersection |
| `IntraMAT_COMP_mask.nii.gz` | Intersection |
| `Inter-&IntraMAT_mask.nii.gz` | Intersection |

All output masks match the original image dimensions and can be directly overlaid on the source scans.

### Volume spreadsheets (saved in APP_FOLDER)

| File | Contents |
|------|----------|
| `WHOLEMUSCLE_SAT_volumes.xlsx` | SAT and whole muscle volumes per scan |
| `THIGH_MUSCLECOMP_volumes.xlsx` | Thigh compartment volumes per scan |
| `CALF_MUSCLECOMP_volumes.xlsx` | Calf compartment volumes per scan |
| `MAT_volumes.xlsx` | MAT, InterMAT, IntraMAT volumes per scan |

## Stage Dependencies

```
Model A ─────────────> MAT quantification
   |                        |
   v                        v
 Bridge ──> Model B ──> IMAT quantification
        ──> Model C ──> IMAT quantification
```

- **Model A** must run before any other stage (or its output masks must already exist).
- **Bridge** runs automatically when Model B or C is enabled.
- **MAT** requires Model A output (`WHOLEMUSCLE_SAT_mask.nii.gz`).
- **IMAT** requires Model B/C output (`MUSCLECOMP_mask.nii.gz`).

You can re-run later stages without re-running earlier ones, as long as the required output files from previous stages exist in the subject folders.

## Hardware

- **GPU (CUDA)**: Recommended. The pipeline auto-detects CUDA and uses it when available.
- **CPU**: Supported but slower. Use `torch>=2.6.0,<2.10.0` to avoid a known Conv3d crash on CPU.
