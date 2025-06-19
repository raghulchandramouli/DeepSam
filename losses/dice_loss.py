# A customized implementation of both Dice Loss and BCE Loss. is implemented

import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.0):
    """
    Comnpute Dice Loss:
    Args:
        pred (Tensor) : Raw logits from model.
        Target (Tensor) : Ground truth labels.
        smootgh (float) : Smoothing factor to avoid division by zero.
        
    Returns:
        Tensor: Computed Dice Loss.
    """
    
    
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

class DiceBCELoss(nn.Module):
    """
    A combined loss function that includes both Dice Loss and Binary Cross-Entropy Loss.
    """
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        dice = dice_loss(pred, target)
        bce = self.bce(pred, target)
        return dice + bce
    
def iou_score(pred, target, threshold=0.5, eps=1e-6):
    """
    Compute Intersection over Union (IoU) score.
    
    Args:
        pred (Tensor): Raw logits from model.
        target (Tensor): Ground truth labels.
        threshold (float): Threshold to binarize predictions.
        eps (float): Small value to avoid division by zero.
        
    Returns:
        float: Computed IoU score.
    """
    
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    
    return iou.mean()