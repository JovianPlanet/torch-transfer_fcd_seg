import torch
from torch import nn
from torchmetrics import Dice


def dice_coeff(x, y, smooth=1e-6):

    #comment out if your model contains a sigmoid or equivalent activation layer
    #x = F.sigmoid(x)       
    
    #flatten label and prediction tensors
    x = x.reshape(-1)#view(-1)
    y = y.reshape(-1)#view(-1)

    if y.sum() == 0 and x.any():
        #print(f'sum labels = {y.sum()}')
        x = (x==0) #torch.where(x==0, 1, 0)
        y = (y==0)
    
    intersection = (x * y).sum()                            
    dice = torch.mean((2.*intersection + smooth)/(x.sum() + y.sum() + smooth))  
        
    return dice

#PyTorch
#https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = nn.Sigmoid()  
        inputs = preds(inputs) 

        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)#view(-1)
        targets = targets.reshape(-1)#view(-1)

        if targets.sum() == 0 and inputs.sum() > 0:
            inputs = torch.where(inputs==0, 1, 0)#(inputs==0) 
            targets = torch.where(targets==0, 1, 0)#(targets==0)
        
        intersection = (inputs * targets).sum()                            
        dice = torch.mean((2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth))  
        
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, weight = None, size_average = True, device='device'):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss() # Usar el dice de este codigo
        #self.dice = Dice(zero_division=1, ignore_index=0).to(device) # Usar el dice de torchmetrics

    def forward(self, inputs, targets, smooth = 1):
        assert(inputs.shape == targets.shape)
        # dice_loss = 1. - self.dice(inputs, targets.long()) # Para el dice de la libreria torchmetrics
        dice_loss = self.dice(inputs, targets)
        bce_loss = self.bce(inputs, targets)

        return  dice_loss + bce_loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = nn.CrossEntropyLoss()(inputs, targets)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky