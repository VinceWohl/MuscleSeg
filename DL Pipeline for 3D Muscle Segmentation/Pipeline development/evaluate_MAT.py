from scipy.ndimage import label
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os 

    
thigh_dict = {
        1: 'Rectus Femoris',
        2: 'Vastus Lateralis',
        3: 'Vastus Intermedius',
        4: 'Vastus Medialis',
        5: 'Sartorius',
        6: 'Gracilis',
        7: 'Biceps Femoris',
        8: 'Semitendinosus',
        9: 'Semimembranosus',
        10: 'Adductor Brevis',
        11: 'Adductor Longus',
        12: 'Adductor Magnus',
        13: 'Guteus Maximus', 
        }

calf_dict = {
        1: 'Gastrocnemius Medialis',
        2: 'Gastrocnemius Lateralis',
        3: 'Soleus',
        4: 'Flexor Digitorum Longus',
        5: 'Flexor Hallucis Longus',
        6: 'Tibialis Posterior',
        7: 'Peroneus',
        8: 'Tibialis Anterior',
        9: 'Extensor Longus'}


def load_nifi_volume(ni_path):
    return nib.load(ni_path).get_fdata()

def compute_MAT(fat_path, whole_muscle_mask_path, muscle_comp_mask_path, limb, save_mask=False):
    
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
    # compute total mat volume     
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
    
    # compute inter mat volume 
    inter_mat_volume = np.sum(inter_mat_img) * np.prod(pixel_dim)
    
    # compute muscle specific Intra-MAT
    intra_mat_volumes = {}
    muscles_dic = thigh_dict if limb == "THIGH" else calf_dict
    
    for label_key, muscle_name in muscles_dic.items():
        tmp_muscle_mask = np.where(muscle_comp_mask == label_key,1, 0)
        tmp_intra_mat_img = fat_img * tmp_muscle_mask
         # compute muscle specific Intra-MAT volume
        intra_mat_volumes[muscle_name] = np.sum(tmp_intra_mat_img) * np.prod(pixel_dim)
    
    return total_mat_volume, inter_mat_volume, intra_mat_volumes


def compute_muscle_volumes(whole_muscle_mask_path, muscle_comp_mask_path, limb):
    
    # ---------------------------------------------------------------------------
    # Load data 
    # ---------------------------------------------------------------------------    
    whole_muscle_mask = load_nifi_volume(whole_muscle_mask_path)
    muscle_comp_mask = load_nifi_volume(muscle_comp_mask_path)
    
    # ---------------------------------------------------------------------------
    # Compute Total MAT volume
    # ---------------------------------------------------------------------------
    # Create Muscle only mask
    whole_muscle_mask = np.where(whole_muscle_mask == 2, 1, 0)
    # Multiply fat_img with mc_mask
    total_mat_img = fat_img * whole_muscle_mask
    # compute total mat volume     
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
    
    # compute inter mat volume 
    inter_mat_volume = np.sum(inter_mat_img) * np.prod(pixel_dim)
    
    # compute muscle specific Intra-MAT
    intra_mat_volumes = {}
    muscles_dic = thigh_dict if limb == "THIGH" else calf_dict
    
    for label_key, muscle_name in muscles_dic.items():
        tmp_muscle_mask = np.where(muscle_comp_mask == label_key,1, 0)
        tmp_intra_mat_img = fat_img * tmp_muscle_mask
         # compute muscle specific Intra-MAT volume
        intra_mat_volumes[muscle_name] = np.sum(tmp_intra_mat_img) * np.prod(pixel_dim)
    
    return total_mat_volume, inter_mat_volume, intra_mat_volumes


# ------------------------------------------------------------------------------------------------------
#                   Help function for compute MAT on predictions
# ------------------------------------------------------------------------------------------------------



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

def find_bbox(wholemuscle):
    labels, num_labels = label(wholemuscle)
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1  # Skip background label 0
    largest_contour_mask = labels == largest_label
    indices = np.where(largest_contour_mask)
    min_x, max_x = np.min(indices[0]), np.max(indices[0])
    min_y, max_y = np.min(indices[1]), np.max(indices[1])
    min_x -= 10
    max_x += 10
    min_y -= 10
    max_y += 10
    return min_x, max_x, min_y, max_y

def get_shape_diff(muscle_mask_path):
    muscle_mask = load_nifi_volume(muscle_mask_path)
    muscle_mask = np.where(muscle_mask==2, 1, 0)
    min_x, max_x, min_y, max_y = find_bbox(muscle_mask)
    return [min_x, muscle_mask.shape[0]-max_x, min_y, muscle_mask.shape[1]-max_y]

def restore_orig_shape(muscle_mask_path, pred):
    shape_diff = get_shape_diff(muscle_mask_path)
    # X_dim
    xa = shape_diff[0]
    xb = shape_diff[1]
    # Y_dim
    ya = shape_diff[2]
    yb = shape_diff[3]
    pred = np.pad(pred, ((xa, xb), (ya, yb), (0, 0)), mode='constant')
    return pred


def reshape_and_copy_pred_masks(fat_path, whole_muscle_mask_path, muscle_comp_mask_path, gt_muscle_mask_path, tmp):
    
    # ---------------------------------------------------------------------------
    # Load Fat Volume 
    # ---------------------------------------------------------------------------
    fat_nii = nib.load(fat_path)
    pixel_dim = fat_nii.header['pixdim'][1:4]
    fat_img = fat_nii.get_fdata()
    # Normalize values in fat_img to the range [0, 1]
    fat_img = (fat_img - np.min(fat_img)) / (np.max(fat_img) - np.min(fat_img))
    # thresholding
    fat_img = np.where(fat_img < 0.1, 0, 1)    
    
    # ---------------------------------------------------------------
    # load masks and reshape it to fit original fat image
    whole_muscle_mask = retransform(load_nifi_volume(whole_muscle_mask_path), fat_nii.shape)
    muscle_comp_mask = load_nifi_volume(muscle_comp_mask_path)
    
    # restore to bounding box shape
    ref_mask = load_nifi_volume(tmp)
    ref_shape = ref_mask.shape 
    muscle_comp_mask_b = retransform(muscle_comp_mask, [ref_shape[0], ref_shape[1]])

    # back to original image shape 
    gt_muscle_mask = load_nifi_volume(gt_muscle_mask_path)
    gt_muscle_mask = np.where(gt_muscle_mask==2, 1, 0)
    min_x, max_x, min_y, max_y = find_bbox(gt_muscle_mask)   
    xa, xb, ya, yb = [min_x, fat_img.shape[0]-max_x, min_y, fat_img.shape[1]-max_y]
    muscle_comp_mask_b = np.pad(muscle_comp_mask_b, ((xa, xb), (ya, yb), (0, 0)), mode='constant')
    
    # Extract filename from the file path
    nib.save(nib.Nifti1Image(whole_muscle_mask, affine=fat_nii.affine),
                 os.path.join(os.path.dirname(fat_path),
                              os.path.basename(whole_muscle_mask_path).replace("cropped_", "")))
    
    nib.save(nib.Nifti1Image(muscle_comp_mask_b, affine=fat_nii.affine, header=fat_nii.header),
                os.path.join(os.path.dirname(fat_path),
                            os.path.basename(muscle_comp_mask_path).replace("cropped_", "")))
    

def reshape_predicted_masks(data_path, data_path_pred_masks, dst_path):
        
    for subject in tqdm(os.listdir(data_path)):
        subject_path = data_path/subject    
        
        for limb in ["CALF", "THIGH"]:
            subject_limb_path = subject_path/limb
            reshape_and_copy_pred_masks(
                subject_limb_path/"roi_70p_Fat.nii.gz",
                data_path_pred_masks / f"{subject}_{limb}_WHOLEMUSCLE_SAT_pred.nii.gz", 
                data_path_pred_masks / f"{subject}_{limb}_MUSCLE_COMP_cropped_pred.nii.gz",
                subject_limb_path/ f"{subject}_{limb}_WHOLEMUSCLE_SAT_mask.nii.gz",
                subject_limb_path/ f"{subject}_{limb}_MUSCLE_COMP_cropped_mask.nii.gz")
            


def compute_MAT_GT(data_path, dst_path, pred=False):
    
    results_list = []
    
    # Initialize empty lists to store the results for the current subject
    calf_results_list = []
    thigh_results_list = []
    
    for subject in tqdm(os.listdir(data_path)):
        subject_path = data_path/subject    
        
        for limb in ["CALF", "THIGH"]:
            subject_limb_path = subject_path/limb
             # compute whole muscle MAT and Muscle specific IntraMAT
            
            if pred:
                wholemuscle_mask_path = subject_limb_path/ (subject + "_" + limb + "_WHOLEMUSCLE_SAT_pred.nii.gz")
                muscle_comp_mask_path = subject_limb_path/(subject + "_" + limb + "_MUSCLE_COMP_pred.nii.gz")
            else: 
                wholemuscle_mask_path = subject_limb_path/(subject + "_" + limb + "_WHOLEMUSCLE_SAT_mask.nii.gz")
                muscle_comp_mask_path = subject_limb_path/ (subject + "_" + limb + "_MUSCLE_COMP_mask.nii.gz")
            
            total_mat_volume, inter_mat_volume, intra_mat_volumes = compute_MAT(subject_limb_path/"roi_70p_Fat.nii.gz",
                                     wholemuscle_mask_path, muscle_comp_mask_path, limb, False)
            
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
    dst_calf = dst_path/"calf_mat_pred_volumes.xlsx" if pred else dst_path/"calf_mat_gt_volumes.xlsx"
    dst_thigh = dst_path/"thigh_mat_pred_volumes.xlsx" if pred else dst_path/"thigh_mat_gt_volumes.xlsx"
    pd.concat(calf_results_list, axis=0, ignore_index=True).to_excel(dst_calf)
    pd.concat(thigh_results_list, axis=0, ignore_index=True).to_excel(dst_thigh)             
        

def main():
    
    # -------------------------
    # Compute mat on GT
    # -------------------------
    data_path_gt = Path(r"/media/yeshe/Expansion/Work_during_PhD/Projects/Interns/Vinent/04 Python scripts/00 Pipeline development/AIPS Data 20th April 2022")
    dst_path = Path(r"/media/yeshe/Expansion/Work_during_PhD/Projects/Interns/Vinent/04 Python scripts/00 Pipeline development/MAT_results")
    # compute_MAT_GT(data_path_gt, dst_path)
        
    # -------------------------
    # Copy and reshape Predicted masks
    # -------------------------
    # data_path_pred_masks = Path(r"/media/yeshe/Expansion/Work_during_PhD/Projects/Interns/Vinent/TMP/Model_A_all_predictions")
    # reshape_predicted_masks(data_path_gt, data_path_pred_masks, dst_path)

    # -------------------------
    # Compute mat on GT
    # -------------------------
    compute_MAT_GT(data_path_gt, dst_path, True)




if __name__ == "__main__":
    main()