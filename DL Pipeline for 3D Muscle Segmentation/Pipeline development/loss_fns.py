# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:14:42 2023

@author: tp-vincentw

This script defiens the loss function.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
for TRAINING
'''
def softdice(c_pred, c_target, smooth=1e-7):   # class prediction and class target
    intersection = (c_target*c_pred).sum()                            
    softdice = (2.*intersection + smooth)/(c_pred.sum() + c_target.sum() + smooth) 
    return softdice

class BCEDiceLoss(nn.Module):    
    def __init__(self, num_classes, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.softdice = softdice
        self.num_classes = num_classes
        
    def forward(self, pred, target):
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        
        DSC = 0.0
        BCE = 0.0
        for class_ in range(self.num_classes):
            # prepare prediction and target
            c_pred = pred[:, class_, :, : , :]
            c_target = target[: , class_, :, : , :]
            c_pred = c_pred.contiguous().view(-1)
            c_target = c_target.contiguous().view(-1)
            # compute dice loss 
            DSC += 1 - self.softdice(c_pred, c_target)
            # compute bce loss 
            BCE += F.binary_cross_entropy_with_logits(c_pred, c_target, reduction='mean')    
        BCE_DSC = BCE + DSC
        
        return BCE_DSC

def binary_focal_loss_with_logits(pred, target, gamma=2):
    # compute the binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    
    # compute the focal term: (1 - p)^gamma, where p is the predicted probability
    p = torch.sigmoid(pred)
    focal_term = torch.pow(1 - p, gamma)
    
    # Compute the final focal loss
    focal_loss = focal_term * loss
    
    # Return the average focal loss
    return focal_loss.mean()

class BFDiceLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, weight=None, size_average=True):
        super(BFDiceLoss, self).__init__()
        self.softdice = softdice
        self.num_classes = num_classes
        self.gamma=gamma
        
    def forward(self, pred, target):
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
            
        DSC = 0.0
        BF = 0.0
        for class_ in range(self.num_classes):
            # prepare prediciton and target
            c_pred = pred[:, class_, :, : , :]
            c_target = target[: , class_, :, : , :]
            c_pred = c_pred.contiguous().view(-1)
            c_target = c_target.contiguous().view(-1)
            # compute dice loss 
            DSC += 1 - self.softdice(c_pred, c_target)
            # compute binary focal loss 
            BF += binary_focal_loss_with_logits(c_pred, c_target, gamma=self.gamma)
        BF_DSC = BF + DSC
            
        return BF_DSC

'''
for TESTING
'''
def dice(c_pred, c_target):   # class prediction and class target
    c_pred = (c_pred > 0.5)
    intersection = (c_target*c_pred).sum()                            
    dice = (2*intersection)/(c_pred.sum() + c_target.sum()) 
    return dice

class DiceScore(nn.Module):    
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()
        self.dice = dice
        
    def forward(self, dic, pred, target):
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        
        DSC = 0.0
        for class_id, class_ in dic.items():
            # prepare prediction and target
            c_pred = pred[:, class_id, :, :, :]
            c_target = target[:, class_id, :, :, :]
            c_pred = c_pred.contiguous().view(-1)
            c_target = c_target.contiguous().view(-1)
            # compute dice loss for the class
            class_['score'] = self.dice(c_pred, c_target)
            DSC += class_['score']
            
        return dic, DSC/len(dic)