# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:07:42 2023

@author: tp-vincentw

This script conducts the evaluation of the acquired segmentation masks of Model A, B and C
Following metrics are included:
    - false-postive rate                    FPR = FP/(TP+FN) in percent
    - false-negative rate                   FNR = FN/(TP+FN) in percent
    - volume difference                     Diff = (V(GT)-V(P))/V(GT) in percent
    - housedorff distance 95% percentile
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from medpy import metric
import matplotlib.pyplot as plt
import seg_metrics.seg_metrics as sg
import re

def get_files_A(path):
    thigh_gts = []
    thigh_prds = []
    calf_gts = []
    calf_prds = []
    
    for i in range(1,6):
        folder = path + f'/v{i}/test_results'
        for file in os.listdir(folder):
            if 'THIGH' in file:
                if 'mask' in file:        
                    thigh_gts.append(folder + '/' + file)
                if 'pred' in file:
                    thigh_prds.append(folder + '/' + file)
            if 'CALF' in file:
                if 'mask' in file:        
                    calf_gts.append(folder + '/' + file)
                if 'pred' in file:
                    calf_prds.append(folder + '/' + file)
    return thigh_gts, thigh_prds, calf_gts, calf_prds


def get_files_B(path):
    gts = []
    prds = []
    
    for i in range(1,6):
        folder = path + f'/v{i}/test_results'
        for file in os.listdir(folder):
            if 'mask' in file:        
                gts.append(folder + '/' + file)
            if 'pred' in file:
                prds.append(folder + '/' + file)
    return gts, prds


def get_files_C(path):
    gts = []
    prds = []
    
    for i in range(1,6):
        folder = path + f'/v{i}/test_results'
        for file in os.listdir(folder):
            if 'mask' in file:        
                gts.append(folder + '/' + file)
            if 'pred' in file:
                prds.append(folder + '/' + file)
    return gts, prds


def determine_class_metrics(class_id, gt, prd, header):

    n_gt = np.where(gt != class_id, 1, 0)
    n_prd = np.where(prd != class_id, 1, 0)
    p_gt = np.where(gt == class_id, 1, 0)
    p_prd = np.where(prd == class_id, 1, 0)
    
    # dice similary coefficient 
    dsc = 2*np.sum(p_gt*p_prd)/np.sum(p_gt+p_prd)
    
    # calc_HausDorff_Distance
    hd = metric.hd(p_prd, p_gt, voxelspacing=(header['pixdim'][1], header['pixdim'][2], header['pixdim'][3]), connectivity=1)
    
    # calc_HausDorff_Distance
    hd95 = metric.hd95(p_prd, p_gt, voxelspacing=(header['pixdim'][1], header['pixdim'][2], header['pixdim'][3]), connectivity=1)
    
    # volumes in cm^3
    v_gt = (np.sum(p_gt) * header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3]) / 1000
    v_prd = (np.sum(p_prd) * header['pixdim'][1]*header['pixdim'][2]*header['pixdim'][3]) / 1000
    
    # differences in percent
    diff = 100*(v_gt-v_prd)/v_gt

    # FPR and FNR in percent
    fp = p_prd - p_gt
    fp[fp<0] = 0
    fpr = 100*np.sum(fp)/np.sum(p_gt)

    fn = n_prd - n_gt
    fn[fn<0] = 0
    fnr = 100*np.sum(fn)/np.sum(p_gt)
    
    # housedorff distance 95 percentile
    #metrics = sg.write_metrics(labels=[1],
    #                  gdth_img=p_gt,
    #                  pred_img=p_prd,
    #                  metrics='hd95')
    # hd95 = metrics[0]['hd95'][0]*((header['pixdim'][1]+header['pixdim'][2]+header['pixdim'][3])/3)  # taking the mean spacing as mean distance per voxel in mm^3
    
    return dsc, v_gt, v_prd, diff, fpr, fnr, hd, hd95


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    CI_low    = md - 1.96*sd
    CI_high   = md + 1.96*sd

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    return md, sd, mean, CI_low, CI_high


def get_number_from_str(string):
    return re.findall(r'\d+', string)[0]



def evaluate_A(data_path, DIC, get_files_fn):
    
    volume_dic = {}
    cw_matrix = []
    head_row = ['Label class',
                'DSC mean', 'DSC sd',
                'V(GT) mean /cm^3', 'V(GT) sd /cm^3',
                'V(P) mean /cm^3', 'V(P) sd /cm^3',
                'FPR mean /%', 'FPR sd /%',
                'FNR mean /%', 'FNR sd /%',
                'Diff. mean /%', 'Diff sd /%',
                'hd mean /mm', 'hd sd /mm',
                'hd95 mean /mm', 'hd95 sd /mm',
                ]
    cw_matrix.append(head_row)
    thigh_gts, thigh_prds, calf_gts, calf_prds = get_files_fn(data_path)
    # fig = 0 # for Bland-Altman plots
    
    '''
    # thigh
    ##########################################
    '''
    print('THIGH')
    cw_matrix.append(['THIGH', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ' ])
    for class_id, class_ in DIC.items():
        print(class_id, class_['name'])
        new_row = [f"{class_['name']}"]
        
        t_dsc_scores = []
        thigh_gt_volumes = []
        thigh_prd_volumes = []
        t_fprs = []
        t_fnrs = []
        t_diffs = []
        t_hds = []
        t_hd95s = []
        sub_id_thigh = []
        
        for point in zip(thigh_gts, thigh_prds):
            gt = nib.load(point[0])            
            header = gt.header.copy()
            gt = gt.get_fdata().astype(np.uint)
            prd = nib.load(point[1])
            prd = prd.get_fdata().astype(np.uint)
            
            # determine metrics 
            dsc, v_gt, v_prd, diff, fpr, fnr, hd, hd95 = determine_class_metrics(class_id, gt, prd, header)

            t_dsc_scores.append(dsc)
            thigh_gt_volumes.append(v_gt)
            thigh_prd_volumes.append(v_prd)
            t_diffs.append(diff)
            t_fprs.append(fpr)
            t_fnrs.append(fnr)
            t_hds.append(hd)
            t_hd95s.append(hd95)
            sub_id_thigh.append(get_number_from_str(str(os.path.basename(point[0]))))
            
        # save class specific evaluation results
                # Create a DataFrame
        df = pd.DataFrame({
            't_dsc_scores': t_dsc_scores,
            'thigh_gt_volumes': thigh_gt_volumes,
            'thigh_prd_volumes': thigh_prd_volumes,
            't_fprs': t_fprs,
            't_fnrs': t_fnrs,
            't_diffs': t_diffs,
            't_hds': t_hds,
            't_hd95s': t_hd95s})
        # Compute mean and standard deviation
        mean_row = df.mean()
        std_row = df.std()
        # Create a single row with mean and standard deviation
        mean_std_row = pd.Series({col: f'{mean_row[col]:.4f} ± {std_row[col]:.4f}' for col in mean_row.index}, name='Mean ± SD')
        # Concatenate the original DataFrame with the mean, std, and mean ± std rows
        #df = pd.concat([df, mean_row.rename('Mean'), std_row.rename('Standard Deviation'), mean_std_row])
        df = pd.concat([df, mean_std_row.to_frame().T], ignore_index=True)
        # Add the 'sub_id_thigh' column to the DataFrame
        df = pd.concat([df, pd.Series(sub_id_thigh, name='sub_id_thigh')], axis=1)
        file_name = class_['name'] + "_thigh_evaluation.xlsx" 
        df.to_excel(os.path.join(data_path, file_name), index=False)
            
        # new_row.extend([np.mean(t_dsc_scores), np.std(t_dsc_scores),
        #                 np.mean(thigh_gt_volumes), np.std(thigh_gt_volumes), 
        #                 np.mean(thigh_prd_volumes), np.std(thigh_prd_volumes),
        #                 np.mean(t_fprs), np.std(fprs),
        #                 np.mean(fnrs), np.std(fnrs),
        #                 np.mean(diffs), np.std(diffs),
        #                 np.mean(hds), np.std(hds),
        #                 np.mean(hd95s), np.std(hd95s)
        #                 ])
        # cw_matrix.append(new_row)
        
        volume_dic.update({"subjct_id": sub_id_thigh})
        volume_dic.update({'thigh_' + class_['name'] + "_gt": thigh_gt_volumes})
        volume_dic.update({'thigh_' + class_['name']+ "_pred": thigh_prd_volumes})
        
        
        # # display Bland-Altman plots
        # fig +=1
        # plt.figure(fig)
        # md, sd, mean, CI_low, CI_high = bland_altman_plot(thigh_gt_volumes, thigh_prd_volumes)
        # plt.title(r" Mean Difference: $\mathbf{Thigh}$ - " + class_['name'])
        # plt.xlabel("Means (cm\u00B3)")
        # plt.ylabel("Difference in cm\u00B3 (GT-P)")
        # plt.ylim(md - 3.5*sd, md + 3.5*sd)
        
        # xOutPlot = np.min(mean) + (np.max(mean)-np.min(mean))*1.14
        
        # plt.text(xOutPlot, md - 1.96*sd, 
        #     r'-1.96SD:' + "\n" + "%.2f" % CI_low, 
        #     ha = "center",
        #     va = "center",
        #     )
        # plt.text(xOutPlot, md + 1.96*sd, 
        #     r'+1.96SD:' + "\n" + "%.2f" % CI_high, 
        #     ha = "center",
        #     va = "center",
        #     )
        # plt.text(xOutPlot, md, 
        #     r'Mean:' + "\n" + "%.2f" % md, 
        #     ha = "center",
        #     va = "center",
        #     )
        # plt.subplots_adjust(right=0.85)
        # plt.show()
    
    '''
    calf
    ##########################################
    '''
    print('CALF')
    cw_matrix.append(['CALF', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ' ])
    for class_id, class_ in DIC.items():
        print(class_id, class_['name'])
        new_row = [f"{class_['name']}"]
        
        c_dsc_scores = []
        calf_gt_volumes = []
        calf_prd_volumes = []
        c_fprs = []
        c_fnrs = []
        c_diffs = []
        c_hds = []
        c_hd95s = []
        sub_id_calf = []
        
        for point in zip(calf_gts, calf_prds):
            gt = nib.load(point[0])
            header = gt.header.copy()
            gt = gt.get_fdata().astype(np.uint)
            prd = nib.load(point[1])
            prd = prd.get_fdata().astype(np.uint)
            
            # determine metrics 
            dsc, v_gt, v_prd, diff, fpr, fnr, hd, hd95 = determine_class_metrics(class_id, gt, prd, header)

            c_dsc_scores.append(dsc)
            calf_gt_volumes.append(v_gt)
            calf_prd_volumes.append(v_prd)
            c_diffs.append(diff)
            c_fprs.append(fpr)
            c_fnrs.append(fnr)
            c_hds.append(hd)
            c_hd95s.append(hd95)
            sub_id_calf.append(get_number_from_str(str(os.path.basename(point[0]))))

        # new_row.extend([np.mean(dsc_scores), np.std(dsc_scores),
        #                 np.mean(calf_gt_volumes), np.std(calf_gt_volumes), 
        #                 np.mean(calf_prd_volumes), np.std(calf_prd_volumes),
        #                 np.mean(fprs), np.std(fprs),
        #                 np.mean(fnrs), np.std(fnrs),
        #                 np.mean(diffs), np.std(diffs),
        #                 np.mean(hds), np.std(hds),
        #                 np.mean(hd95s), np.std(hd95s)
        #                 ])
        # cw_matrix.append(new_row)
        
        # --------------------------------------------
        # save class specific metics results 
        df_c = pd.DataFrame({
            'c_dsc_scores': c_dsc_scores,
            'calf_gt_volumes': calf_gt_volumes,
            'calf_prd_volumes': calf_prd_volumes,
            'c_fprs': c_fprs,
            'c_fnrs': c_fnrs,
            'c_diffs': c_diffs,
            'c_hds': c_hds,
            'c_hd95s': c_hd95s})
        # Compute mean and standard deviation
        mean_row = df_c.mean()
        std_row = df_c.std()
        # Create a single row with mean and standard deviation
        mean_std_row = pd.Series({col: f'{mean_row[col]:.4f} ± {std_row[col]:.4f}' for col in mean_row.index}, name='Mean ± SD')
        # Concatenate the original DataFrame with the mean, std, and mean ± std rows
        df_c = pd.concat([df_c, mean_std_row.to_frame().T], ignore_index=True)
        # Add the 'sub_id' column
        df_c = pd.concat([df_c, pd.Series(sub_id_calf, name='sub_id_calf')], axis=1)
        file_name = class_['name'] + "_calf_evaluation.xlsx"
        df_c.to_excel(os.path.join(data_path, file_name), index=False)
            
        # volume_dic.update({"subject_id": sub_id_calf})
        volume_dic.update({'calf_' + class_['name']+ "_gt": calf_gt_volumes})
        volume_dic.update({'calf_' + class_['name']+ "_pred": calf_prd_volumes})
        
        
        # # display Bland-Altman plots
        # fig +=1
        # plt.figure(fig)
        # md, sd, mean, CI_low, CI_high = bland_altman_plot(calf_gt_volumes, calf_prd_volumes)
        # plt.title(r" Mean Difference: $\mathbf{Calf}$ - " + class_['name'])
        # plt.xlabel("Means (cm\u00B3)")
        # plt.ylabel("Difference in cm\u00B3 (GT-P)")
        # plt.ylim(md - 3.5*sd, md + 3.5*sd)
        
        # xOutPlot = np.min(mean) + (np.max(mean)-np.min(mean))*1.14
        
        # plt.text(xOutPlot, md - 1.96*sd, 
        #     r'-1.96SD:' + "\n" + "%.2f" % CI_low, 
        #     ha = "center",
        #     va = "center",
        #     )
        # plt.text(xOutPlot, md + 1.96*sd, 
        #     r'+1.96SD:' + "\n" + "%.2f" % CI_high, 
        #     ha = "center",
        #     va = "center",
        #     )
        # plt.text(xOutPlot, md, 
        #     r'Mean:' + "\n" + "%.2f" % md, 
        #     ha = "center",
        #     va = "center",
        #     )
        # plt.subplots_adjust(right=0.85)
        # plt.show()
        

    # ====================================================================
    # save volumes for bland altman plot
    # ====================================================================
    # pd.DataFrame(volume_dic).to_excel(f'{data_path}/volumes.xlsx', index=False)

def remove_slashes_and_spaces(input_string):
    # Remove slashes
    result_string = input_string.replace('/', '')
    # Remove spaces
    result_string = result_string.replace(' ', '')
    return result_string


def evaluate_BnC(data_path, DIC, get_files_fn, model):

    volume_dic = {}
    cw_matrix = []
    head_row = ['Label class',
                'DSC mean', 'DSC sd',
                'V(GT) mean /cm^3', 'V(GT) sd /cm^3',
                'V(P) mean /cm^3', 'V(P) sd /cm^3',
                'FPR mean /%', 'FPR sd /%',
                'FNR mean /%', 'FNR sd /%',
                'Diff. mean /%', 'Diff sd /%',
                'hd mean /mm', 'hd sd /mm',
                'hd95 mean /mm', 'hd95 sd /mm',
                ]
    cw_matrix.append(head_row)
    
    gts, prds = get_files_fn(data_path)
    
    for class_id, class_ in DIC.items():
        print(class_id, class_['name'])
        new_row = [f"{class_['name']}"]
        
        dsc_scores = []
        volumes_gt = []
        volumes_prd = []
        fprs = []
        fnrs = []
        diffs = []
        hds = []
        hd95s = []
        sub_id = []
        
        for point in zip(gts,prds):
            gt = nib.load(point[0])
            header = gt.header.copy()
            gt = gt.get_fdata().astype(np.uint)
            prd = nib.load(point[1])
            prd = prd.get_fdata().astype(np.uint)
            
            # determine metrics 
            dsc, v_gt, v_prd, diff, fpr, fnr, hd, hd95 = determine_class_metrics(class_id, gt, prd, header)

            dsc_scores.append(dsc)
            volumes_gt.append(v_gt)
            volumes_prd.append(v_prd)
            diffs.append(diff)
            fprs.append(fpr)
            fnrs.append(fnr)
            hds.append(hd)
            hd95s.append(hd95)
            sub_id.append(get_number_from_str(str(os.path.basename(point[0]))))
    
        new_row.extend([np.mean(dsc_scores), np.std(dsc_scores),
                        np.mean(volumes_gt), np.std(volumes_gt), 
                        np.mean(volumes_prd), np.std(volumes_prd),
                        np.mean(fprs), np.std(fprs),
                        np.mean(fnrs), np.std(fnrs),
                        np.mean(diffs), np.std(diffs),
                        np.mean(hds), np.std(hds),
                        np.mean(hd95s), np.std(hd95s)
                        ])
        cw_matrix.append(new_row)
        
        volume_dic.update({"subject_id": sub_id})
        volume_dic.update({class_['name']+ "_gt": volumes_gt})
        volume_dic.update({class_['name']+ "_pred": volumes_prd})
        
        # save class specific evaulation results
        df = pd.DataFrame({
            'dsc_scores': dsc_scores,
            'gt_volumes': volumes_gt,
            'prd_volumes': volumes_prd,
            'fprs': fprs,
            'fnrs': fnrs,
            'diffs': diffs,
            'hds': hds,
            'hd95s': hd95s})
        
        # Compute mean and standard deviation
        mean_row = df.mean()
        std_row = df.std()
        # Create a single row with mean and standard deviation
        mean_std_row = pd.Series({col: f'{mean_row[col]:.4f} ± {std_row[col]:.4f}' for col in mean_row.index}, name='Mean ± SD')
        # Concatenate the original DataFrame with the mean, std, and mean ± std rows
        df = pd.concat([df, mean_std_row.to_frame().T], ignore_index=True)
        # Add the 'sub_id_thigh' column to the DataFrame
        df = pd.concat([df, pd.Series(sub_id, name='sub_id')], axis=1)
        
        file_name = remove_slashes_and_spaces(class_['name'] + "_"+ model + "_evaluation.xlsx")
        df.to_excel(os.path.join(data_path,file_name), index=False)        
            
        
        # # display Bland-Altman plots
        # plt.figure(class_id)
        # md, sd, mean, CI_low, CI_high = bland_altman_plot(volumes_gt, volumes_prd)
        # plt.title(f"Mean Difference: {class_['name']}")
        # plt.xlabel("Means (cm\u00B3)")
        # plt.ylabel("Difference in cm\u00B3 (GT-P)")
        # plt.ylim(md - 3.5*sd, md + 3.5*sd)
        
        # xOutPlot = np.min(mean) + (np.max(mean)-np.min(mean))*1.14
        
        # plt.text(xOutPlot, md - 1.96*sd, 
        #     r'-1.96SD:' + "\n" + "%.2f" % CI_low, 
        #     ha = "center",
        #     va = "center",
        #     )
        # plt.text(xOutPlot, md + 1.96*sd, 
        #     r'+1.96SD:' + "\n" + "%.2f" % CI_high, 
        #     ha = "center",
        #     va = "center",
        #     )
        # plt.text(xOutPlot, md, 
        #     r'Mean:' + "\n" + "%.2f" % md, 
        #     ha = "center",
        #     va = "center",
        #     )
        # plt.subplots_adjust(right=0.85)
        # plt.show()
    
    # ====================================================================    
    # save mean evaluation results    
    # ====================================================================
        
    # ====================================================================
    # save volumes for bland altman plot
    # ====================================================================
    # pd.DataFrame(volume_dic).to_excel(f'{data_path}/volumes.xlsx', index=False)


#########################################################################################################################
#########################################################################################################################



main_path = '/media/yeshe/Expansion/Work_during_PhD/Projects/Interns/Vinent/04 Python scripts/00 Pipeline development' 

'''
Model A: SAT and Whole Muscle eval.
###################################################################################################################################################################
'''
print('Evaluate Model A')
path_A = os.path.join(main_path, 'Model_A_results/Model_A_exp_17 - final')
DIC_A = {
        1: {'name': 'SAT'           },
        2: {'name': 'Whole Muscle'  }
        }
# evaluate_A(path_A, DIC_A, get_files_A) 

'''
Model B: Thigh Muscle Compartments eval.
###################################################################################################################################################################
'''
print('Evaluate Model B')
path_B = os.path.join(main_path, 'Model_B_results/Model_B_exp_11 - final')
DIC_B = {
        1: {'name': 'Rectus Femoris'                        },
        2: {'name': 'Vastus Lateralis'                      },
        3: {'name': 'Vastus Intermedius'                    },
        4: {'name': 'Vastus Medialis'                       },
        5: {'name': 'Sartorius'                             },
        6: {'name': 'Gracilis'                              },
        7: {'name': 'Biceps Femoris'                        },
        8: {'name': 'Semitendinosus'                        },
        9: {'name': 'Semimembranosus'                       },
        10: {'name': 'Adductor Brevis'          },
        11: {'name': 'Adductor Longus'                      },
        12: {'name': 'Adductor Magnus'  },
        13: {'name': 'Gluteus Maximus'                      }
        }
evaluate_BnC(path_B, DIC_B, get_files_B, "Thigh")


'''
Model C: Calf Muscle Compartments eval.
###################################################################################################################################################################
'''
print('Evaluate Model C')
path_C = os.path.join(main_path,'Model_C_results/Model_C_exp_14 - final')
DIC_C = {
        1: {'name': 'Gastrocnemius Medialis'                },
        2: {'name': 'Gastrocnemius Lateralis'               },
        3: {'name': 'Soleus'                                },
        4: {'name': 'Flexor Digitorum Longus'               },
        5: {'name': 'Flexor Hallucis Longus'                },
        6: {'name': 'Tibialis Posterior'                    },
        7: {'name': 'Peroneus Longus and Brevis'            },
        8: {'name': 'Tibialis Anterior'                     },
        9: {'name': 'Digitorum Longus'  }
        }
evaluate_BnC(path_C, DIC_C, get_files_C, "Calf")   
