import argparse
import os
import scipy.misc
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from load_road import make_data_splits
from torchvision import models
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
from torchsummary import summary
from model import SegNet
from model_leaky import SegNet_leaky
from model_deep import SegNet_deep
from torch import no_grad
from torch import FloatTensor
from torch import save
from torch import load
import matplotlib.pyplot as plt
from model_atrous import SegNet_atrous
from DeepLabv3_plus import DeepLabv3_plus
from model_atrous_nl import SegNet_atrous_nl
from model_atrous_multi import SegNet_atrous_multi

def test(base_dir, batch_size):
        # Define Dataloader
    train_loader, val_loader, test_loader, nclass = make_data_splits(base_dir, batch_size=4)

        # Define network
    with open(os.path.join(os.path.join(base_dir, 'test.txt')), "r") as f:
            lines = f.read().splitlines()    

    model = SegNet_atrous_multi(4,2)
    model = model.cuda()
    model.load_state_dict(load("/home/ahana/pytorch_road/models/SegNet_best_atrous_4c_n_multi_lr_001"))
    model.eval()
    tbar = tqdm(test_loader)
    overall_confusion_matrix = None
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()
        with no_grad():
            output_ce, output_dice, output_iou = model(image)

        pred_ce = output_ce.data.cpu().numpy()
        pred_dice = output_dice.data.cpu().numpy()
        pred_iou = output_iou.data.cpu().numpy()
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
    test(base_dir, batch_size)


if __name__ == "__main__":
   main()
