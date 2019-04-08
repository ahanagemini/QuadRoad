import argparse
import os
import scipy.misc
import numpy as np
import sys
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from load_road.load_road_4c import make_data_splits_4c
from load_road.load_road_3c import make_data_splits_3c
from load_road.load_road_1c import make_data_splits_1c
from torchvision import models
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
from torchsummary import summary
from models.model import SegNet
from models.model_leaky import SegNet_leaky
from torch import no_grad
from torch import FloatTensor
from torch import save
import matplotlib.pyplot as plt
from models.model_deep import SegNet_deep
from models.model_atrous import SegNet_atrous
from models.DeepLabv3_plus import DeepLabv3_plus
from models.model_leaky import SegNet_leaky
from models.model_atrous_nl import SegNet_atrous_nl
from losses.losses import DiceLoss
from losses.losses import FocalLoss
from losses.losses import IoULoss
from models.model_atrous_multi import SegNet_atrous_multi

'''
A code to execute train for minimization of sum of 3 losses using multi-task approach to train each loss:
    cross-entropy, IoU and Soft.
    Args: num_channels, num_classes, file_prefix
          num_channels: number of input channels
          num_classes: how many classes to be predicted
          norm: Whether to use normalization or not 0 or 1.
          file_prefix: prefix used to name the trained models and the result files
          validation will not work for 17 classes and does not fit on 1080 GPU
'''
def training_and_val(epochs, base_dir, batch_size, num_channels, num_class, norm, file_prefix):
    # Define Dataloader
    if num_class == 17:
        cat_dir = 'ground_truth_500'
    if num_class == 2:
        cat_dir = 'rev_annotations'
    if num_channels == 4:
        train_loader, val_loader, test_loader, nclass = make_data_splits_4c(base_dir, num_class, cat_dir, norm, batch_size=4)
    if num_channels == 3:
        train_loader, val_loader, test_loader, nclass = make_data_splits_3c(base_dir, num_class, cat_dir, norm, batch_size=4)
    if num_channels == 1:
        train_loader, val_loader, test_loader, nclass = make_data_splits_1c(base_dir, num_class, cat_dir, norm, batch_size=4)
    if num_channels == 8:
        train_loader, val_loader, test_loader, nclass = make_data_splits_hs(base_dir, num_class, cat_dir, norm, batch_size=4)
    if num_channels == 0: # for using with the 4 predictions
        train_loader, val_loader, test_loader, nclass = make_data_splits_p(base_dir, num_class, cat_dir, norm, batch_size=4)
        num_channels = 4
    # Define network
    
    model = SegNet_atrous_multi(num_channels,num_class)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    weights = [0.29, 1.69]
    class_weights = FloatTensor(weights).cuda()
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_iou = IoULoss()
    criterion_dice = DiceLoss()
    criterion_iou = criterion_iou.cuda()
    criterion_dice = criterion_dice.cuda()
    criterion_ce = criterion_ce.cuda()
        
        # Define lr scheduler
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            # args.epochs, len(self.train_loader))

    # Using cuda
    model = model.cuda()
    best = 0.0
    losses = []
    ious = []
    best_epoch = 0
    # Start training epochs
    for epoch in range(0, epochs):
        # Training
        train_loss = 0.0
        model.train()
        tbar = tqdm(train_loader)
        num_img_tr = len(train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            optimizer.zero_grad()
            output_ce, output_dice, output_iou = model(image)
            loss_iou = criterion_iou(output_iou, target)
            loss_ce = criterion_ce(output_ce, target)
            loss_dice = criterion_dice(output_dice, target)
            loss = loss_iou + loss_ce + loss_dice
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print('Train loss: %.3f' % (train_loss / (i + 1)))
        print('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        losses.append(train_loss)
        # Validation 
        model.eval()
        tbar = tqdm(val_loader)
        test_loss = 0.0
        overall_confusion_matrix = None
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            with no_grad():
                output_ce, output_dice, output_iou = model(image)
            loss_iou = criterion_iou(output_iou, target)
            loss_ce = criterion_ce(output_ce, target)
            loss_dice = criterion_dice(output_dice, target)
            loss = loss_iou + loss_ce + loss_dice
            #loss = criterion(output, target)
            test_loss += loss.item()
            pred_iou = output_iou.data.cpu().numpy()
            pred_dice = output_dice.data.cpu().numpy()
            pred_ce = output_ce.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = pred_iou + pred_dice + pred_ce
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            target_f = target.flatten()
            pred_f = pred.flatten()
            current_confusion_matrix = confusion_matrix(y_true=target_f, y_pred=pred_f, labels=[0, 1])
        
            if overall_confusion_matrix is not None:
                overall_confusion_matrix += current_confusion_matrix
            else:
                overall_confusion_matrix = current_confusion_matrix
    
        #Compute IoU
        intersection = overall_confusion_matrix[1][1]
        ground_truth_set = overall_confusion_matrix.sum(axis=1)
        predicted_set = overall_confusion_matrix.sum(axis=0)
        union =  overall_confusion_matrix[0][1] + overall_confusion_matrix[1][0]

        intersection_over_union = intersection / union.astype(np.float32)
        RMIoU = intersection/(ground_truth_set + predicted_set - intersection)
        # Save trained models
        if RMIoU[1] > best:
            best = RMIoU[1]
            best_epoch = epoch
            #save(model.state_dict(), "/home/ahana/pytorch_road/model_best/SegNet_best_5em0")
            save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/best_"+file_prefix)
        if epoch % 10 == 0:
            #save(model.state_dict(), "/home/ahana/pytorch_road/model_best/SegNet_pre_atrous_5em0")
            outfile = "/home/ahana/pytorch_road/trained_models/"+file_prefix+"_" + str(epoch)
            save(model.state_dict(), outfile)

        if epoch == 75:
            outfile = "/home/ahana/pytorch_road/trained_models/"+file_prefix+"_75"
            save(model.state_dict(), outfile)

        #save(model.state_dict(), "/home/ahana/pytorch_road/model_best/SegNet_final_atrous_5em0")
        save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/final_"+file_prefix)
        ious.append(RMIoU[1])
        # Fast test during the training
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
        print("RMIoU: {}, Intersection: {}, Ground truth: {}, Predicted: {}, Best: {}".format(RMIoU, intersection, ground_truth_set, predicted_set, best))

    # Save results
    losses1 = [(i,x) for i,x in enumerate(losses)]
    ious1 = [(i,x) for i,x in enumerate(ious)]
    fp = open("/home/ahana/pytorch_road/result_texts/"+file_prefix+".txt", "w")
    fp.write(str(losses1))
    fp.write("\n")
    fp.write(str(ious1))
    fp.write("\n")
    fp.write(str(best_epoch))
    fp.close()
    
    plt.plot(losses)
    plt.savefig("/home/ahana/pytorch_road/loss_graphs/loss_"+file_prefix+".png")
    plt.clf()
    plt.plot(ious)
    print(losses)
    print(ious)
    plt.savefig("/home/ahana/pytorch_road/iou_graphs/iou_"+file_prefix+".png")    

def main():

    base_dir = "/home/ahana/road_data"
    epochs = 80 
    batch_size = 4
    print('Starting Epoch: 0')
    print('Total Epoches:', epochs)
    num_channels = int(sys.argv[1])
    num_class = int(sys.argv[2])
    norm = int(sys.argv[3])
    file_prefix = sys.argv[4]
    training_and_val(epochs, base_dir, batch_size, num_channels, num_class, norm, file_prefix)


if __name__ == "__main__":
   main()
