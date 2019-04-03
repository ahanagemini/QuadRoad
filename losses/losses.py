import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
Code to support multiple losses:
    Dice, IoU and Focal
    TODO: Dice and IoU need to be modified to support > 2 classes

'''

class FocalLoss(nn.Module):

    def __init__(self, weights, gamma=0, alpha=1):
        super(FocalLoss, self).__init__()
        self.weights=weights
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target):

        cross_entropy = F.cross_entropy(output, target, self.weights)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt

        balanced_focal_loss = self.alpha * focal_loss

        return balanced_focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        iflat = input[:,1]
        tflat = target.view(-1)
        tflat = tflat.float()
        intersection = (iflat * tflat).sum()
    
        return 1 - ((2. * intersection + self.smooth) /
              (iflat.sum() + tflat.sum() + self.smooth))

class IoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        iflat = input[:,1]
        tflat = target.view(-1)
        tflat = tflat.float()
        intersection = (iflat * tflat).sum()

        return 1 - ((intersection + self.smooth) /
              (iflat.sum() + tflat.sum() - intersection + self.smooth))

class NonBinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(NonBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        num_classes = 17
        input = torch.sigmoid(input)
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1) #flatten
        #print(target.type())
        # One-hot encode
        target = target.reshape(target.size(0),1)
        target = torch.zeros(len(target), 17).cuda().scatter_(1, target, 1)
        #target = one_hot_target
        print(target.shape)
        print(input.shape)
        intersection = torch.sum(input * target, dim=0)
        denominator = torch.sum(input, dim=0) + torch.sum(target, dim=0)
        return -1 * torch.sum(2. * intersection + self.smooth / (denominator + self.smooth))
   

class NonBinaryIoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(NonBinaryIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1) # flatten
        # one-hot encode
        target = target.reshape(target.size(0),1)
        #one_hot_target = (target == torch.arange(num_classes).reshape(1,num_classes)).float()
        target = torch.zeros(len(target), 17).cuda().scatter_(1, target, 1)
        print(target.shape)
        print(input.shape)
        intersection = torch.sum(input * target, dim=0)
        denominator = torch.sum(input, dim=0) + torch.sum(target, dim=0) - intersection
        return -1 * torch.sum(intersection + self.smooth / (denominator + self.smooth))

