# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:19:42 2023

@author: tp-vincentw

This scripts trains a model to segment the muscle compartments in calf MR images.
"""

import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import (
    get_files_C,
    splitup,
    get_loaders,
    validate,
    model_application,
    save_scores
    )
from transform_fns import get_transforms_C
from dataset import MRI_Dataset
from model_C import UNET
from loss_fns import (BFDiceLoss, DiceScore)

# Hyperparamter etc.
EXP = 14
NUM_FOLDS = 1
NUM_EPOCHS = 4500
PATIENCE = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
IMAGE_WIDTH = 160   # original 384
IMAGE_HEIGHT = 144  # original 288
IMAGE_DEPTH = 48    # original 45-68
PIN_MEMORY = True
TEST_MODEL = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = 'AIPS Data 20th April 2022'
DIC = {
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
NUM_CLASSES = len(DIC)

# free GPU memory and reset seed
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
torch.manual_seed(0)

print(f'Model_C - EXP: {EXP}')
print(f'NUM_FOLDS: {NUM_FOLDS}, NUM_EPOCHS: {NUM_EPOCHS}, PATIENCE: {PATIENCE}, BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}, NUM_WORKERS: {NUM_WORKERS}')
print(f'IMG_SHAPE (W, H, D): {IMAGE_WIDTH}, {IMAGE_HEIGHT}, {IMAGE_DEPTH}\n')


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    t_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = target.to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            probmap = model(data)
            loss = loss_fn(probmap, target)
            t_loss += loss.item()
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    return t_loss/len(loader)


def main():
    for fold in range(1,NUM_FOLDS+1):
        print(f'--------------------------------------- MODEL_Cv{fold} STARTS ---------------------------------------')
        start_time = time.time()
        model_path = f'Model_C_exp_{EXP}/v{fold}'
        os.makedirs(model_path, exist_ok=True)
        
        # get data directions and partition them
        IMGS, MASKS = get_files_C(DATA_PATH)
        TRAIN_IMGS, TRAIN_MASKS, VAL_IMGS, VAL_MASKS, TEST_IMGS, TEST_MASKS = splitup(IMGS, MASKS, model_path, fold)
                
        # get transform functions
        TRAIN_TF, VAL_TF, TEST_TF = get_transforms_C(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)
        
        # get the dataloaders for training and validation
        train_loader, val_loader = get_loaders(
            TRAIN_IMGS,
            TRAIN_MASKS,
            VAL_IMGS,
            VAL_MASKS,
            
            NUM_CLASSES,
            
            TRAIN_TF,
            VAL_TF,
            
            BATCH_SIZE,
            
            num_workers = NUM_WORKERS,
            pin_memory = PIN_MEMORY
            )
        
        # initialize separate test dataset
        test_ds = MRI_Dataset(TEST_IMGS, TEST_MASKS, NUM_CLASSES, transform=TEST_TF)
        
        print(f'N train: {len(TRAIN_MASKS)}, N val: {len(VAL_MASKS)}, N test: {len(TEST_MASKS)}')
        
        # initialize model architecture
        model = UNET(in_channels=2, out_channels=NUM_CLASSES).to(device=DEVICE)
        
        # define loss function for training and validation and test_fn for testing
        loss_fn = BFDiceLoss(num_classes=NUM_CLASSES, gamma=3)
        test_fn = DiceScore() # defining num_classes not required cause dynamic
        
        # define optimizer algorithm
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        
        # define scaler
        scaler = torch.cuda.amp.GradScaler()
       
        print('------------------------------------------------------------------------')
        print(f'MODEL_Cv{fold}: Training')
        check_path = f'{model_path}/checkpoints'
        os.makedirs(check_path, exist_ok=True)
        train_loss = []
        val_loss = []
        best_val = [100, 0]
        pat = 0
        
        for epoch in range(NUM_EPOCHS):
            print(f'=> Ep.{epoch}')
            
            # train model
            t_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
            train_loss.append(t_loss)
            
            # validate model
            v_loss = validate(model, val_loader, loss_fn, device=DEVICE)
            val_loss.append(v_loss)
            
            # check if validaiton loss is the best, display losses and save checkpoint
            if v_loss < best_val[0]:
                best_val[0] = v_loss
                best_val[1] = epoch
                pat = 0
                
                print(f'Train loss: {t_loss} - Best val loss: {best_val[0]}')
                
                # save model
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }
                checkpoint_path = f"{check_path}/cp_model_Cv{fold}_epoch_{epoch}.pth"
                torch.save(checkpoint, checkpoint_path)
            
            else: pat += 1
            if pat > PATIENCE: break
            
        save_scores(train_loss, val_loss, best_val, model_path)
        print(f'Best validation loss: {best_val[0]} in epoch: {best_val[1]}')
        
        # display the running time
        end_time = time.time()
        running_time_minutes = (end_time - start_time)//60
        hours = int(running_time_minutes // 60)
        minutes = int(running_time_minutes % 60)
        print(f'Runtime for training Model Cv{fold}: {hours}:{minutes}')
        print('------------------------------------------------------------------------')
        
        if TEST_MODEL:
            print(f'MODEL_Cv{fold}: Testing')
            test_folder = f'{model_path}/test_results'
            os.makedirs(test_folder,exist_ok=True)
            
            # load the model with the best validation result
            checkpoint = torch.load(f'{check_path}/cp_model_Cv{fold}_epoch_{best_val[1]}.pth', map_location=torch.device(DEVICE))
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'=> Best checkpoint loaded: cp_model_Cv{fold}_epoch_{best_val[1]}.pth')
            
            # apply model for testing
            model_application(model, DIC, test_ds, test_fn, IMAGE_DEPTH, test_folder, mID='C', device=DEVICE)
            
        print(f'--------------------------------------- MODEL_Cv{fold} COMPLETE ---------------------------------------')     
     
   
if __name__ == "__main__":
    main()