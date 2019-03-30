import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#class FocalLoss(nn.Module):
#    def __init__(self, gamma=0, size_average=False):
#        super(FocalLoss, self).__init__()
#        self.gamma = gamma
#        self.size_average = size_average

#    def forward(self, input, target):
#        if input.dim()>2:
#            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#        target = target.view(-1,1)
#
#        logpt = F.log_softmax(input)
#        logpt = logpt.gather(1,target)
#        logpt = logpt.view(-1)
#        pt = Variable(logpt.data.exp())


#        loss = -1 * (1-pt)**self.gamma * logpt
#        if self.size_average: return loss.mean()
#        else: return loss.sum()

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

#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#import torch.autograd as autograd
#from torch.autograd import Variable
#import matplotlib.pyplot as plt
#import random
#from tqdm import tqdm_notebook as tqdm
#import math

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

#class FocalLoss(nn.Module):
#    def __init__(self, gamma):
#        super().__init__()
#        self.gamma = gamma
##        
#    def forward(self, input, target):
#        # Inspired by the implementation of binary_cross_entropy_with_logits
#        if not (target.size() == input.size()):
#            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
##
#        max_val = (-input).clamp(min=0)
#        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
#
#       # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
#        invprobs = F.logsigmoid(-input * (target * 2 - 1))
#        loss = (invprobs * self.gamma).exp() * loss
#        
#        return loss.mean()

