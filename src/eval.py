import argparse
import os
import scipy.misc
import numpy as np
import sys
from tqdm import tqdm
from torchvision import models
from torch import nn
from torch import optim
from torchsummary import summary
from torch import no_grad
from torch import FloatTensor
from torch import save
from torch import load
import matplotlib.pyplot as plt
from load_road import *
from models.model import SegNet
from models.model_A import SegNet_A
from models.model_AL import SegNet_AL
from models.model_hs import SegNet_hs
from models.model_hs_A import SegNet_hs_A
from models.model_hs_AL import SegNet_hs_AL
from metrics.iou import IoU

'''
A code to execute evaluation for the specified model
    Usage: eval.py --batch_size <batch_size>
                --num_channels <num_channels> --num_class <num_class>
                --model_name <model saved filename> --model <model_name>
                --split <data split> [--norm]

          num_channels: 1, 3, 4, 8
          num_class: 2 or 17
          model: model may be SegNet, SegNet_A, SegNet_AL, SegNet_hs,
            SegNet_hs_A, SegNet_hs_AL, SegNet_hs_ALD
          norm: whether to perform normalization or not. boolean
          model_name: filename for file to load saved model from
          split: train, valid or test
'''

def data_loader(num_channels, base_dir,
        num_class, norm, purpose, batch_size, split):
    """
    Function that loads data for all splits based on num_channels
    Args:
        num_channels: number of channels in input data
        base_dir: directory that contains data
        num_class: number of classes
        norm: whether to perform normalization
        purpose: train/test
        batch_size: batch size for trainig
        split: The split to be used
    Returns:
        data_loader for required split
    """
    if num_channels == 4:
        train_loader, val_loader, test_loader = rgb_hght.split_data(base_dir,
                num_class, norm, 'eval', batch_size=4)
    elif num_channels == 3:
        train_loader, val_loader, test_loader = rgb.split_data(base_dir,
                num_class, norm, 'eval', batch_size=4, augment=False)
    elif num_channels == 1:
        train_loader, val_loader, test_loader = hght.split_data(base_dir,
                num_class, norm, 'eval', batch_size=4, augment=False)
    elif num_channels == 8:
        train_loader, val_loader, test_loader = hs.split_data(base_dir,
                num_class, norm, 'eval', batch_size=4, augment=False)
    else:
        print("Number of channels not supported")
        sys.exit()

    if split == 'train':
        return train_loader
    if split == 'valid':
        return val_loader
    if split == 'test':
        return test_loader

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

def create_save_paths(base_dir, model, model_name, num_class, num_channels):
    """
    Function that creates paths for saving predictions
    softmax probailities (softmax in range 0-200) for road class
    Args:
        base_dir: directory for data and results
        model: model type
        model_name: saved model name
        num_class: number of classes
        num_channels: number of channels in input data
    Returns:
        pred_dir path and heat_dir path
    """
    save_dir = f'{model}_{model_name}_{num_class}_{num_channels}'
    pred_dir = f'{base_dir}/preds/pred_{save_dir}'
    heat_dir = f'{base_dir}/heatmaps/heat_{save_dir}'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(heat_dir):
        os.makedirs(heat_dir)
    return pred_dir, heat_dir

def test(base_dir, batch_size, num_channels, num_class,
        norm, model_name, split, model):
    """
    Function to perform evaluation and save th epredictions
    and softmax probabilities (softmax probabilities 
    for road class only in range 0-200)
    Args:
        base_dir: directory that cocntains all data
        batch_size: the training batch size
        num_channels
        num_class
        norm: Use normalization or not
        model_name: name of saved model file
        model: the name of the model
        split: train, valid, test
    """

    # Define Dataloader
    split_loader = data_loader(num_channels, base_dir,
            num_class, norm, 'train', batch_size, split)
    # List of  file names depending on split
    with open(os.path.join(os.path.join(base_dir, f'{split}.txt')), "r") as f:
        lines = f.read().splitlines()
    pred_dir, heat_dir = create_save_paths(base_dir, model, model_name,
            num_class, num_channels)
    # Define and load network
    model = select_model(model, num_channels, num_class)
    model = model.cuda()
    model.load_state_dict(load(f'{base_dir}/trained_models/{model_name}'))
    # Start tests
    model.eval()
    metric = IoU(num_class)
    tbar = tqdm(split_loader)
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()
        with no_grad():
            output = model(image)
        pred = output.data.cpu()
        target = target.cpu()
        metric.add(pred, target)
        # Code to get predictions
        pred_save = np.argmax(pred, axis=1)
        # Code to use for saving output heat maps
        fn = nn.Softmax()
        sftmx = fn(output)
        heat_save = sftmx.data.cpu().numpy()
        for j in range(0, batch_size):
            predFilepath = f'{pred_dir}/{lines[i*batch_size+j]}.png'
            pred = pred_save[j,:,:]
            scipy.misc.toimage(pred,cmin=0, cmax=255).save(predFilepath)
            heat = heat_save[j,1,:,:]
            heat = heat * 200.0
            padded_heat = heat.astype(dtype=np.uint8)
            unpad_heat = padded_heat[6:506,6:506]
            heatFilepath = f'{heat_dir}/{lines[i*batch_size+j]}.png'
            scipy.misc.toimage(unpad_heat, cmin=0,cmax=255).save(heatFilepath)

    iou, miou = metric.value()
    if num_class == 17:
        miou = (miou*17 - iou[0])/16
    print("IoU: {}, MIoU: {}, RMIoU: {}".format(iou, miou, iou[1]))
    metric.reset()

def main():

    cwd = os.getcwd()
    base_dir = f'{cwd}/data'
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--batch_size', type=int, default='4',
                    help='batch size for training')
    parser.add_argument('--num_channels', type=int, default=3,
                    help='Number of input channels')
    parser.add_argument('--num_class', type=int, default='2',
                    help='Number of classes')
    parser.add_argument('--model', type=str, default='SegNet',
                    help='Name of model to be used')
    parser.add_argument('--model_name', type=str, default='test',
                    help='exact filename of model to load')
    parser.add_argument('--norm', action='store_true',
                    help='whether to use normalization')
    parser.add_argument('--split', type=str, default='test',
                    help='split to be used for evaluation')
    args = parser.parse_args()
    test(base_dir, args.batch_size, args.num_channels, args.num_class,
            args.norm, args.model_name, args.split, args.model)


if __name__ == "__main__":
   main()
