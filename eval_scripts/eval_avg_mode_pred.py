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
from load_road.load_hs import make_data_splits_hs
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
from models.model_atrous_hs import SegNet_atrous_hs
from models.model_shallow import SegNet_shallow
from models.model_atrous_GN import SegNet_atrous_GN
'''
A code to execute tests using models fro rgb, hght, hs and 17class to 2 class predictions 
and give the softmax sum max as the predicted class:
    We can use any 2 to 4 of models based on the arg provided
    Args: num_channels, num_classes, model_name_ce, model_name_dice, model_name_iou
          num_channels: number of input channels
          num_classes: how many classes to be predicted
          model_name_rgb: name of trained model to load for rgb, 'unused' if this mdoel is not used
          model_name_hght: name of trained model to load for hght, 'unused' if this mdoel is not used
          model_name_hs: name of trained model to load for hs, 'unused' if this mdoel is not used
          model_name_p: name of trained model for shallow network 17 class to 2 class, 'unused' if this mdoel is not used
    TODO: Test this GPU was in use
'''

def test(base_dir, batch_size, model_name_rgb, model_name_hght, model_name_hs, model_name_p):
#def test(base_dir, batch_size, model_name_rgb, model_name_hght):
 # Define Dataloader
    if model_name_rgb != 'unused':
        train_loader_rgb, val_loader_rgb, test_loader_rgb, nclass = make_data_splits_3c(base_dir, batch_size=4)
    if model_name_hght != 'unused':
        train_loader_hght, val_loader_hght, test_loader_hght, nclass = make_data_splits_1c(base_dir, batch_size=4, directory='hght')
    if model_name_hs != 'unused':
        train_loader_hs, val_loader_hs, test_loader_hs, nclass = make_data_splits_hs(base_dir, batch_size=4)
    if model_name_p != 'unused':
        train_loader_p, val_loader_p, test_loader_p, nclass = make_data_splits_1c(base_dir, batch_size=4, directory='pred_17_4c')
    # List of test file names

    # List test file names
    with open(os.path.join(os.path.join(base_dir, 'test.txt')), "r") as f:
            lines = f.read().splitlines()    

    # Define and load models
    if model_name_rgb != 'unused':
        model_rgb = SegNet_atrous_GN(3,2)
    if model_name_hght != 'unused':
        model_hght = SegNet_atrous_GN(1, 2)
    if model_name_hs != 'unused':
        model_hs = SegNet_atrous_hs(8,2)
    if model_name_p != 'unused':
        model_p17 = SegNet_shallow(1,2)

    if model_name_rgb != 'unused':
        model_rgb = model_rgb.cuda()
        model_rgb.load_state_dict(load("/home/ahana/pytorch_road/trained_models/"+model_name_rgb))
        model_rgb.eval()
        tbar_rgb = tqdm(test_loader_rgb)
        enum_rgb = enumerate(tbar_rgb)
    if model_name_hght != 'unused':
        model_hght = model_hght.cuda()
        model_hght.load_state_dict(load("/home/ahana/pytorch_road/trained_models/"+model_name_hght))
        model_hght.eval()
        tbar_hght = tqdm(test_loader_hght)
        enum_hght = enumerate(tbar_hght)
    if model_name_hs != 'unused':
        model_hs = model_hs.cuda()
        model_hs.load_state_dict(load("/home/ahana/pytorch_road/trained_models/"+model_name_hs))
        model_hs.eval()
        tbar_hs = tqdm(test_loader_hs)
        enum_hs = enumerate(tbar_hs)
    if model_name_p != 'unused':    
        model_p17 = model_p17.cuda()
        model_p17.load_state_dict(load("/home/ahana/pytorch_road/trained_models/"+model_name_p))
        model_p17.eval()
        tbar_p = tqdm(test_loader_p)
        enum_p = enumerate(tbar_p)
    # Perform Test
    overall_confusion_matrix = None
    for i, sample in enumerate(tbar_rgb):
        if model_name_rgb != 'unused':
            sample_rgb = next(enum_rgb)
            image_rgb, target_rgb = sample_rgb[1]['image'], sample_rgb[1]['label']
            image_rgb, target_rgb = image_rgb.cuda(), target_rgb.cuda()
        if model_name_hght != 'unused':
            sample_hght = next(enum_hght)
            image_hght, target_hght = sample_hght[1]['image'], sample_hght[1]['label']
            image_hght, target_hght = image_hght.cuda(), target_hght.cuda()
        if model_name_hs != 'unused':
            sample_hs = next(enum_hs)
            image_hs, target_hs = sample_hs[1]['image'], sample_hs[1]['label']
            image_hs, target_hs = image_hs.cuda(), target_hs.cuda()
        if model_name_p != 'unused':
            sample_p = next(enum_p)
            image_p, target_p = sample_p[1]['image'], sample_p[1]['label']
            image_p, target_p = image_p.cuda(), target_p.cuda()
        with no_grad():
            fn = nn.Softmax2d()
            if model_name_rgb != 'unused':
                output_rgb = model_rgb(image_rgb)
            if model_name_hght != 'unused':
                output_hght = model_hght(image_hght)
            if model_name_hs != 'unused':    
                output_hs = model_hs(image_hs)
            if model_name_p != 'unused':    
                output_p = model_p17(image_p)

        if model_name_rgb != 'unused':
            output_rgb = fn(output_rgb)
        if model_name_hght != 'unused':
            output_hght = fn(output_hght)
        if model_name_hs != 'unused':
            output_hs = fn(output_hs)
        if model_name_p != 'unused':
            output_p = fn(output_p)

        avg_lst = []
        if model_name_rgb != 'unused':
            pred_rgb = output_rgb.data.cpu().numpy()
            avg_lst.append(pred_rgb)
        if model_name_rgb != 'unused':
            pred_hght = output_hght.data.cpu().numpy()
            avg_lst.append(pred_hght)
        if model_name_rgb != 'unused':
            pred_hs = output_hs.data.cpu().numpy()
            avg_lst.append(pred_hs)
        if model_name_rgb != 'unused':
            pred_p = output_p.data.cpu().numpy()
            avg_lst.append(pred_p)

        pred = numpy.zeros(avg_list[0].shape)
        for i in range(0,len(avg_lst)):
            pred = numpy.add(pred,avg_lst[i])
        
        #pred = pred_rgb + pred_hght
        target_rgb = target_rgb.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        target_f = target_rgb.flatten()
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
    model_name_rgb = sys.argv[1]
    model_name_hght = sys.argv[2]
    model_name_hs = sys.argv[3]
    model_name_p17 = sys.argv[4]
    test(base_dir, batch_size, model_name_rgb, model_name_hght, model_name_hs, model_name_p17)
    #test(base_dir, batch_size, model_name_rgb, model_name_hght)


if __name__ == "__main__":
   main()
