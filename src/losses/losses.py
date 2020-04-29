import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
Code to support multiple losses:
    Dice, IoU and Focal for binary and multi-class classification
"""

class FocalLoss(nn.Module):
    """
    Class for computing the focal loss
    """
    def __init__(self, weights, gamma=0, alpha=1):
        """
        :param weights: weight of the classes
        :param gamma: a parameter for focal loss
        :param alpha: a parameter for focal loss
        """

        super(FocalLoss, self).__init__()
        self.weights=weights
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target):
        """
        Function to use output and target to find focal loss
        Args:
            output: predicted segmentation
            target: ground truth segmentation
        """
        cross_entropy = F.cross_entropy(output, target, self.weights)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt

        balanced_focal_loss = self.alpha * focal_loss

        return balanced_focal_loss

class DiceLoss(nn.Module):
    """
    Class for computing the binary Dice loss
    """
    def __init__(self, smooth=1.0):
        """
        :param smooth: a parameter for Dice loss
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        """
        Function to use output and target to find Dice loss
        Args:
            output: predicted segmentation
            target: ground truth segmentation
        """
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
    """
    Class for computing the binary IoU loss
    """

    def __init__(self, smooth=1.0):
        """
        :param smooth: a parameter for IoU loss
        """
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        """
        Function to use output and target to find IoU loss
        Args:
            output: predicted segmentation
            target: ground truth segmentation
        """
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
    """
    Class for computing the multi-class Dice loss
    """

    def __init__(self, smooth=1.0):
        """
        :param smooth: a parameter for Dice loss
        """
        super(NonBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        """
        Function to use output and target to find Dice loss
        Args:
            output: predicted segmentation
            target: ground truth segmentation
        """
        num_classes = 17
        input = torch.sigmoid(input)
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1) #flatten
        target = target.reshape(target.size(0),1)
        target = torch.zeros(len(target), 17).cuda().scatter_(1, target, 1)
        intersection = torch.sum(input * target, dim=0)
        denominator = torch.sum(input, dim=0) + torch.sum(target, dim=0)
        return -1 * torch.sum(2. * intersection + self.smooth / (denominator + self.smooth))
   

class NonBinaryIoULoss(nn.Module):
    """
    Class for computing the multi-class IoU loss
    """

    def __init__(self, smooth=1.0):
        """
        :param smooth: a parameter for IoU loss
        """
        super(NonBinaryIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        """
        Function to use output and target to find IoU loss
        Args:
            output: predicted segmentation
            target: ground truth segmentation
        """
        input = torch.sigmoid(input)
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1) # flatten
        target = target.reshape(target.size(0),1)
        target = torch.zeros(len(target), 17).cuda().scatter_(1, target, 1)
        intersection = torch.sum(input * target, dim=0)
        denominator = torch.sum(input, dim=0) + torch.sum(target, dim=0) - intersection
        return -1 * torch.sum(intersection + self.smooth / (denominator + self.smooth))

