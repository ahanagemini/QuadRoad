import argparse
import os
import scipy.misc
import numpy as np
import sys
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from load_road import *
from torchvision import models
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
from torchsummary import summary
from torch import no_grad
from torch import FloatTensor
from torch import save
from torch import load
import matplotlib.pyplot as plt
from models.model import SegNet
from models.model_A import SegNet_A
from models.model_AL import SegNet_AL
from models.model_hs import SegNet_hs
from models.model_hs_A import SegNet_hs_A
from models.model_hs_AL import SegNet_hs_AL
from losses.losses import *
from metrics.iou import IoU

'''
A code to execute train for minimization of a any one of 3 losses:
    cross-entropy, IoU and Dice.
    Usage: train_scripts/train.py --epochs <epochs> --batch_size <batch_size>
                --num_channels <num_channels> --num_class <num_class>
                --loss_type <ce, dice or IoU> --model <model_name>
                --file_prefix <file_prefix> [--norm]
                [--augment]

          num_channels: 1, 3, 4, 8
          num_class: 2 or 17
          model: model may be SegNet, SegNet_A, SegNet_AL, SegNet_hs,
            SegNet_hs_A, SegNet_hs_AL, SegNet_hs_ALD
          norm: whether to perform normalization or not. boolean
          loss_type: 'dice','IoU' or 'ce'
          file_prefix: prefix used to name the trained models and the result files
          augment: whether to use augmentation. Boolean
'''
def data_loader(num_channels, base_dir,
        num_class, norm, purpose, batch_size, augment):
    """
    Function that loads data for all splits based on num_channels
    Args:
        num_channels: number of channels in input data
        base_dir: directory that contains data
        num_class: number of classes
        norm: whether to perform normalization
        purpose: train/test
        batch_size: batch size for trainig
        augment: whether to use data augmentation
    Returns:
        train_loader, val_loader, test_loader
    """
    if num_channels == 4:
        train_loader, val_loader, test_loader = rgb_hght.split_data(base_dir,
                num_class, norm, 'train', batch_size=4)
    elif num_channels == 3:
        train_loader, val_loader, test_loader = rgb.split_data(base_dir,
                num_class, norm, 'train', batch_size=4, augment=augment)
    elif num_channels == 1:
        train_loader, val_loader, test_loader = hght.split_data(base_dir,
                num_class, norm, 'train', batch_size=4, augment=augment)
    elif num_channels == 8:
        train_loader, val_loader, test_loader = hs.split_data(base_dir,
                num_class, norm, 'train', batch_size=4, augment=augment)
    else:
        print("Number of channels not supported")
        sys.exit()
    
    return train_loader, val_loader, test_loader

def select_model(model, num_channels, num_class):
    """
    Function that selects correct model type
    Args:
        model: name of model
        num_channels: number of channels in input data
        num_class: number of classes
    Returns:
        model object
    """

    if model=='SegNet_A':
        model = SegNet_A(num_channels,num_class)
    elif model=='SegNet_AL':
        model = SegNet_AL(num_channels,num_class)
    elif model=='SegNet':
        model = SegNet(num_channels,num_class)
    elif model == 'SegNet_hs':
        model = SegNet_hs(num_channels, num_class)
    elif model == 'SegNet_hs_A':
        model = SegNet_hs_A(num_channels, num_class)
    elif model == 'SegNet_hs_AL':
        model = SegNet_hs_AL(num_channels, num_class, False)
    elif model == 'SegNet_hs_ALD':
        model = SegNet_hs_AL(num_channels, num_class, True)
    else:
        print("Incorrect model name")
        sys.exit()
    return model

def select_loss(loss_type, num_class):
    """
    Function to set the loss based on num_class
    and loss type
    Args:
        loss_type: type of loss Dice, cross-entropy or IoU
        num_class: number of classes
    Returns:
        loss function as criterioni, class weights
    """
    if num_class == 2:
        weights = [0.54, 6.88]
        class_weights = FloatTensor(weights).cuda()
    if num_class == 17:
        weights = [0.0089, 0.0464, 4.715, 1.00, 0.06655, 27.692, 79.604,
                0.273, 3.106, 0.305, 1, 21.071, 0.0806, 0.636,
                1.439, 0.387, 10.614]
        class_weights = FloatTensor(weights).cuda()
    if loss_type == 'dice' and num_class == 2:
        criterion = DiceLoss()
    if loss_type == 'IoU' and num_class == 2:
        criterion = IoULoss()
    if loss_type == 'dice' and num_class > 2:
        criterion = NonBinaryDiceLoss()
    if loss_type == 'IoU' and num_class > 2:
        criterion = NonBinaryIoULoss()
    if loss_type == 'focal':
        criterion = FocalLoss(weights=class_weights,gamma=1, alpha=1)
    if loss_type == 'ce' and num_class == 2:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == 'ce' and num_class > 2:
        criterion = nn.CrossEntropyLoss()
    return criterion, class_weights

def validation(model, val_loader, criterion, metric, num_class, epoch,
        file_prefix, base_dir, best):
    """
    Function to run validation after each epoch
    Args:
        model: Model so far
        val_loader: validation data loader
        criterion: Loss function info
        metric: metric for computing IoU
        num_class: number of classes
        epoch: The current epoch
        file_prefix: file prefix for saving results
        base_dir: base_dir where data is
        best: best IoU till now
    Returns:
        siou: IoU of each class
        miou: mean IoU
        metric: modified after this validation
    """
    model.eval()
    tbar = tqdm(val_loader)
    test_loss = 0.0
    overall_confusion_matrix = None
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()
        with no_grad():
            output = model(image)
        loss = criterion(output, target)
        test_loss += loss.item()
        pred = output.data.cpu()
        target = target.cpu()
        metric.add(pred, target)

    siou, miou = metric.value()
    # Leaving out the background
    if num_class == 17:
        miou = (miou*17 - siou[0])/16
    else:
        miou = siou[0]
        # Save trained models

    save(model.state_dict(), f'{base_dir}/trained_models/final_{file_prefix}')
    print('Validation:')
    print("MIoU: {}, RMIoU: {}, Best: {}".format(miou, siou[1], best))

    metric.reset()
    return metric, siou, miou

def save_info(losses, ious, file_prefix, base_dir):
    """
    Function to save th etraining info
    Args:
        losses: list of losses over epochs
        ious: list of ious over epochs
        file_prefix: part of the file name specific to this training
        base_dir: the directory for all data
    """
    losses_epochs = [(i,x) for i,x in enumerate(losses)]
    ious_epochs = [(i,x) for i,x in enumerate(ious)]
    fp = open(f'{base_dir}/result_texts/{file_prefix}.txt', "w")
    fp.write(str(losses_epochs))
    fp.write("\n")
    fp.write(str(ious_epochs))
    fp.write("\n")
    fp.write(str(best_epoch))
    fp.close()

    plt.plot(losses)
    plt.savefig(f'{base_dir}/train_graphs/loss_{file_prefix}.png')
    plt.clf()
    plt.plot(ious)
    plt.savefig(f'{base_dir}/train_graphs/iou_{file_prefix}.png')


def train_val(epochs, base_dir, batch_size, num_channels, num_class,
        norm, loss_type, file_prefix, model, augment):
    """
    Function to perform training and validation
    Args:
        epochs: number of epochs to train
        base_dir: directory that cocntains all data
        batch_size: the training batch size
        num_channels
        num_class
        norm: Use normalization or not
        loss_type: the type of loss
        file_prefix: part of the file name specific to this training
        model: the name of the model
        augment: whether to use data augmentation
    """
    # Define Dataloader
    train_loader, val_loader, test_loader = data_loader(num_channels, base_dir,
            num_class, norm, 'train', batch_size, augment)
    # Define netwoirk, optimizer and loss
    model = select_model(model, num_channels, num_class)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    criterion, class_weights = select_loss(loss_type, num_class)
    # Using CUDA
    criterion = criterion.cuda()
    model = model.cuda()
    metric = IoU(num_class)
    best = 0.0
    losses = []
    ious = []
    best_epoch = 0
    # Training epochs
    for epoch in range(0, epochs):
        train_loss = 0.0
        model.train()
        tbar = tqdm(train_loader)
        num_img_tr = len(train_loader)
        print(num_img_tr)
        # Training
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print('Train loss: %.3f' % (train_loss / (i + 1)))
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
        losses.append(train_loss)
        # Validation after each epoch
        metric, siou, miou = validation(model, val_loader, criterion,
                metric, num_class, epoch, file_prefix, base_dir, best)
        ious.append(siou[1])
        if siou[1] > best:
            # Road is class label 1
            best = siou[1]
            best_epoch = epoch
            save(model.state_dict(), f'{base_dir}/trained_models/best_{file_prefix}')
            
    # Save training data
    save_info(losses, ious, file_prefix, base_dir)

def main():
    cwd = os.getcwd()
    base_dir = f'{cwd}/data'
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--epochs', type=int, default=80,
                    help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default='4',
                    help='batch size for training')
    parser.add_argument('--num_channels', type=int, default=3,
                    help='Number of input channels')
    parser.add_argument('--num_class', type=int, default='2',
                    help='Number of classes')
    parser.add_argument('--loss_type', type=str, default='ce',
                    help='Loss function for training')
    parser.add_argument('--model', type=str, default='SegNet',
                    help='Name of model to be used')
    parser.add_argument('--file_prefix', type=str, default='test',
                    help='prefix for storing files for the current execution')
    parser.add_argument('--norm', action='store_true',
                    help='whether to use normalization')
    parser.add_argument('--augment', action='store_true',
                    help='whether to use data augmentation')
    args = parser.parse_args()
    train_val(args.epochs, base_dir, args.batch_size,
            args.num_channels, args.num_class,
            args.norm, args.loss_type, args.file_prefix, args.model, args.augment)


if __name__ == "__main__":
   main()
