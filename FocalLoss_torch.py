import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

class FocalLoss(_WeightedLoss):

    def __init__(self, weight=None, g=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        
        self.gamma = g
        self.weight = weight 
        
    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return focal_loss