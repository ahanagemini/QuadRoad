import argparse
import os
import scipy.misc
import numpy as np
import sys
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from load_road.load_road_3c import make_data_splits_3c
from load_road.load_road_1c import make_data_splits_1c
from load_road.load_road_4c import make_data_splits_4c
from torchvision import models
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
from torchsummary import summary
from models.model import SegNet
from models.model_leaky import SegNet_leaky
from models.model_deep import SegNet_deep
from torch import no_grad
from torch import FloatTensor
from torch import save
from torch import load
import matplotlib.pyplot as plt
from models.model_atrous import SegNet_atrous
from models.DeepLabv3_plus import DeepLabv3_plus
from models.model_atrous_nl import SegNet_atrous_nl

'''
A code to execute tests using multi task model for 3 losses and give the softmax sum max as the predicted class:
    cross-entropy, IoU and Soft.
    Args: num_channels, num_classes, model_name_ce, model_name_dice, model_name_iou
          num_channels: number of input channels
          num_classes: how many classes to be predicted
          cat_dir: directory that has the targets
          norm: normalize or not 0 or 1
          model_name: name of trained model to load for cross entropy
'''

def test(base_dir, batch_size, num_channels, num_class, cat_dir, norm, model_name):
    # Define Dataloader
    if num_class == 17:
        cat_dir = 'ground_truth_500'
    if num_class == 2:
        cat_dir = 'rev_annotations'
    if num_channels == 4:
        train_loader, val_loader, test_loader, nclass = make_data_splits_4c(base_dir, num_class, cat_dir, norm, 'eval', batch_size=4)
    if num_channels == 3:
        train_loader, val_loader, test_loader, nclass = make_data_splits_3c(base_dir, num_class, cat_dir, norm, 'eval', batch_size=4)
    if num_channels == 1:
        train_loader, val_loader, test_loader, nclass = make_data_splits_1c(base_dir, num_class, cat_dir, norm, 'eval', batch_size=4)
    # List of test file names
    if num_channels == 8:
        train_loader, val_loader, test_loader, nclass = make_data_splits_hs(base_dir, num_class, cat_dir, norm, 'eval', batch_size=4)
    if num_channels == 0: # for using with the 4 predictions
        train_loader, val_loader, test_loader, nclass = make_data_splits_p(base_dir, batch_size=4)
        num_channels = 4
    # List test file names
    with open(os.path.join(os.path.join(base_dir, 'test.txt')), "r") as f:
            lines = f.read().splitlines()    

    # Define and load models
    model = SegNet_atrous(num_channels, num_class)
    model = model.cuda()
    model.load_state_dict(load("/home/ahana/pytorch_road/trained_models/"+model_name))
    model.eval()
    #Perform test
    #model_focal = model_focal.cuda()
    #model_focal.load_state_dict(load("/home/ahana/pytorch_road/models/SegNet_best_atrous_4c_n_focal_wt_lr_001"))
    #model_focal.eval()
    tbar = tqdm(test_loader)
    overall_confusion_matrix = None
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()
        with no_grad():
            output_ce, output_dice, output_iou = model(image)
            #output_focal = model_focal(image)

        pred_ce = output_ce.data.cpu().numpy()
        pred_dice = output_dice.data.cpu().numpy()
        pred_iou = output_iou.data.cpu().numpy()
        #pred_focal = output_focal.data.cpu().numpy()
        pred = pred_ce + pred_dice + pred_iou
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        target_f = target.flatten()
        pred_f = pred.flatten()
        current_confusion_matrix = confusion_matrix(y_true=target_f, y_pred=pred_f, labels=[0, 1])

        if overall_confusion_matrix is not None:
            overall_confusion_matrix += current_confusion_matrix
        else:
            overall_confusion_matrix = current_confusion_matrix


        intersection = overall_confusion_matrix[1][1]
        ground_truth_set = overall_confusion_matrix.sum(axis=1)
        predicted_set = overall_confusion_matrix.sum(axis=0)
        union =  overall_confusion_matrix[0][1] + overall_confusion_matrix[1][0]

        intersection_over_union = intersection / union.astype(np.float32)
        RMIoU = intersection/(ground_truth_set + predicted_set - intersection) 
    
    print('Validation:')
    print("RMIoU: {}, Intersection: {}, Ground truth: {}, Predicted: {}".format(RMIoU, intersection, ground_truth_set, predicted_set))


def main():

    base_dir = "/home/ahana/road_data"
    batch_size = 4
    num_channels = int(sys.argv[1])
    num_class = int(sys.argv[2])
    cat_dir = sys.argv[3]
    norm = int(sys.argv[4])
    model_name = sys.argv[5]
    test(base_dir, batch_size, num_channels, num_class, cat_dir, norm, model_name)


if __name__ == "__main__":
   main()
