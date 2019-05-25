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
from load_road.load_hs import make_data_splits_hs
from load_road.load_hs_aug import make_data_splits_hs_aug
from load_road.load_road_pred4 import make_data_splits_p4
from load_road.load_road_pred2 import make_data_splits_p2
from load_road.load_road_3c_aug import make_data_splits_3c_aug
from load_road.load_road_1c_aug import make_data_splits_1c_aug
from torchvision import models
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
from torchsummary import summary
from models.model import SegNet
from models.model_leaky import SegNet_leaky
from models.model_shallow import SegNet_shallow
from torch import no_grad
from torch import FloatTensor
from torch import save
from torch import load
import matplotlib.pyplot as plt
from models.model_deep import SegNet_deep
from models.model_atrous import SegNet_atrous
from models.DeepLabv3_plus import DeepLabv3_plus
from models.model_leaky import SegNet_leaky
from models.model_atrous_nl import SegNet_atrous_nl
from models.model_atrous_hs import SegNet_atrous_hs
from models.model_atrous_GN import SegNet_atrous_GN
from models.model_atrous_GN_dropout import SegNet_atrous_GN_dropout
from models.model_atrous_hs_GN_do import SegNet_atrous_hs_GN_dropout
from models.model_atrous_hs_nl import SegNet_atrous_hs_nl
from losses.losses import DiceLoss
from losses.losses import FocalLoss
from losses.losses import IoULoss
from losses.losses import NonBinaryIoULoss
from losses.losses import NonBinaryDiceLoss
from metrics.iou import IoU

'''
A code to execute train for minimization of a any one of 3 losses:
    cross-entropy, IoU and Soft.
    Args: num_channels, num_classes, file_prefix
          num_channels: number of input channels, also used to indicate augmented data.
                        0 for the model that uses 4 predictions
                        5 for 3 channel augmented
                        2 for 1 channel augmented
                        9 for 8 channel augmented
          num_classes: how many classes to be predicted
          norm: whether to perform normalization or not. 0 or 1 if 2 it indicates prediction on preds
          loss_type: 'dice','IoU' or 'ce'
          file_prefix: prefix used to name the trained models and the result files
          Can use all losses for >2 classes and validation for > 2 classes
'''
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter, num_steps):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(optimizer.param_groups[0]['lr'], i_iter, num_steps, 0.9)
    optimizer.param_groups[0]['lr'] = lr

def training_and_val(epochs, base_dir, batch_size, num_channels, num_class, norm, loss_type, file_prefix, model):
    # Define Dataloader
    if num_class == 17:
        cat_dir = 'ground_truth_500'
    if num_class == 2:
        cat_dir = 'rev_annotations'
    if num_class == 2 and (num_channels == 5 or num_channels == 2 or num_channels == 9):
        cat_dir = 'rev_annot_augment'
    if num_class == 17 and (num_channels == 5 or num_channels == 2 or num_channels == 9):
        cat_dir = 'gt_augment'
    train_set = 'orig'
    #if norm == 2:
    #    if num_channels == 4:
    #        train_loader, val_loader, test_loader, nclass = make_data_splits_p4(base_dir, batch_size=4)
    #    if num_channels == 3:
    #        train_loader, val_loader, test_loader, nclass = make_data_splits_3c_aug(base_dir, num_class, cat_dir, norm, 'train', batch_size=4)
    #else:
    if num_channels == 4:
        train_loader, val_loader, test_loader, nclass = make_data_splits_4c(base_dir, num_class, cat_dir, norm, 'train', batch_size=4)
    elif num_channels == 3:
        train_loader, val_loader, test_loader, nclass = make_data_splits_3c(base_dir, num_class, cat_dir, norm, 'train', batch_size=4)
    elif num_channels == 5:
        train_loader, val_loader, test_loader, nclass = make_data_splits_3c_aug(base_dir, num_class, cat_dir, norm, 'train', batch_size=4)
        num_channels = 3
        train_set = 'aug'
    elif num_channels == 2:
        train_loader, val_loader, test_loader, nclass = make_data_splits_1c_aug(base_dir, num_class, cat_dir, norm, 'train', batch_size=4)
        num_channels = 1
        train_set = 'aug'
    elif num_channels == 1:
        train_loader, val_loader, test_loader, nclass = make_data_splits_1c(base_dir, num_class, cat_dir, norm, 'train', batch_size=4)
    elif num_channels == 8:
        train_loader, val_loader, test_loader, nclass = make_data_splits_hs(base_dir, num_class, cat_dir, norm, 'train', batch_size=4)
    elif num_channels == 9:
        train_loader, val_loader, test_loader, nclass = make_data_splits_hs_aug(base_dir, num_class, cat_dir, norm, 'train', batch_size=4)
        num_channels = 8
        train_set = 'aug'
    elif num_channels == 0: # for using with the 4 predictions
        train_loader, val_loader, test_loader, nclass = make_data_splits_p4(base_dir, batch_size=4)
        num_channels = 4
    else:
        print("NUmber of channels not supported")
    # Define network
    if model=='hs':
        model = SegNet_atrous_hs(num_channels,num_class)
    if model=='hs_nl':
        model = SegNet_atrous_hs_nl(num_channels,num_class)
    elif model == 'hs_GN_do':
        model = SegNet_atrous_hs_GN_dropout(num_channels, num_class)
    elif model == 'shallow':
        model = SegNet_shallow(num_channels, num_class)
    elif model == 'GN':
        model = SegNet_atrous_GN(num_channels, num_class)
    elif model == 'GN_dropout':
        model = SegNet_atrous_GN_dropout(num_channels, num_class)
    elif model == 'atrous':
        model = SegNet_atrous(num_channels, num_class)
    elif model == 'DeepLab':
        model = DeepLabv3_plus(num_channels, num_class)
    else: 
        model = SegNet(num_channels,num_class)
   
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr = 0.001)
            # self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    if num_class == 2:
        #weights = [0.54, 6.87]
        weights = [0.29, 1.69]
        class_weights = FloatTensor(weights).cuda()
    if num_class == 17:
        weights = [0.0089, 0.0464, 4.715, 1.00, 0.06655, 27.692, 79.604, 0.273, 3.106, 0.305, 1, 21.071, 0.0806, 0.636, 1.439, 0.387, 10.614]
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
    if loss_type == 'ce' and num_class > 2 or model == 'DeepLab':
        criterion = nn.CrossEntropyLoss() #not using weights for DeepLab or for 17 classes
    criterion = criterion.cuda()
        
    # Define lr scheduler
    # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            # args.epochs, len(self.train_loader))

    # Using cuda
    model = model.cuda()
    if num_class == 17: # got a MEMORY error with 2 class for training but probably will occur for 17 classes too
        metric = IoU(num_classes=17)
    #model.load_state_dict(load("/home/ahana/pytorch_road/trained_models/best_SegNet_atrous_GN_dropout_rgb_dice_25"))
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
            if model == 'DeepLab':
                adjust_learning_rate(optimizer, i, epochs * num_img_tr)
            output = model(image)
            loss = criterion(output, target)
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
                output = model(image)
            loss = criterion(output, target)
            test_loss += loss.item()
            if num_class == 2:
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
                target_f = target.flatten()
                pred_f = pred.flatten()
                current_confusion_matrix = confusion_matrix(y_true=target_f, y_pred=pred_f, labels=[0, 1])
        
                if overall_confusion_matrix is not None:
                    overall_confusion_matrix += current_confusion_matrix
                else:
                    overall_confusion_matrix = current_confusion_matrix
            else:
                pred = output.data.cpu()
                target = target.cpu()
                #pred = np.argmax(pred, axis=1)
                # Add batch sample into evaluator
                metric.add(pred, target)
    
        # Compute IoU
        if num_class == 2:
            intersection = overall_confusion_matrix[1][1]
            ground_truth_set = overall_confusion_matrix.sum(axis=1)
            predicted_set = overall_confusion_matrix.sum(axis=0)
            union =  overall_confusion_matrix[0][1] + overall_confusion_matrix[1][0]

            intersection_over_union = intersection / union.astype(np.float32)
            RMIoU = intersection/(ground_truth_set + predicted_set - intersection)
            # Save trained best models till certain iterations
            if train_set == 'aug':
                if RMIoU[1] > best and epoch <= 25:
                    best = RMIoU[1]
                    best_epoch = epoch
                    save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/best_"+file_prefix+"_25")
                if RMIoU[1] > best and epoch <= 50:
                    best = RMIoU[1]
                    best_epoch = epoch
                    save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/best_"+file_prefix+"_50")
                if RMIoU[1] > best and epoch <= 75:
                    best = RMIoU[1]
                    best_epoch = epoch
                    save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/best_"+file_prefix+"_75")
                if RMIoU[1] > best:
                    best = RMIoU[1]
                    best_epoch = epoch
                    save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/best_"+file_prefix)
            else:
                if RMIoU[1] > best:
                    best = RMIoU[1]
                    best_epoch = epoch
                    save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/best_"+file_prefix)
            # Saves models to see perfromance over time by testing on test set
            if epoch % 10 == 0:
                outfile = "/home/ahana/pytorch_road/trained_models/"+file_prefix+"_" + str(epoch)
                save(model.state_dict(), outfile)

            if epoch == 75:
                outfile = "/home/ahana/pytorch_road/trained_models/"+file_prefix+"_75"
                save(model.state_dict(), outfile)
        
            if train_set == 'aug':
                if epoch == 50:
                    outfile = "/home/ahana/pytorch_road/trained_models/"+file_prefix+"_50"
                    save(model.state_dict(), outfile)

                if epoch == 25:
                    outfile = "/home/ahana/pytorch_road/trained_models/"+file_prefix+"_25"
                    save(model.state_dict(), outfile)

            save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/final_"+file_prefix)
            ious.append(RMIoU[1])
            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
            print("RMIoU: {}, Intersection: {}, Ground truth: {}, Predicted: {}, Best: {}".format(RMIoU, intersection, ground_truth_set, predicted_set, best))
        
        else:
            siou, miou = metric.value()
            # Leaving out the background
            miou = (miou*17 - siou[0])/16
            # Save trained models
            if miou > best:
                best = miou
                best_epoch = epoch
                save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/best_"+file_prefix)
            if epoch % 10 == 0:
                outfile = "/home/ahana/pytorch_road/trained_models/"+file_prefix+"_" + str(epoch)
                save(model.state_dict(), outfile)

            if epoch == 75:
                outfile = "/home/ahana/pytorch_road/trained_models/"+file_prefix+"_75"
                save(model.state_dict(), outfile)

            save(model.state_dict(), "/home/ahana/pytorch_road/trained_models/final_"+file_prefix)
            ious.append(siou[1])
            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
            print("MIoU: {}, RMIoU: {}, Best: {}".format(miou, siou[1], best))

            metric.reset()

            
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
    num_channels = int(sys.argv[1])
    num_class = int(sys.argv[2])
    norm = int(sys.argv[3])
    loss_type = sys.argv[4]
    file_prefix = sys.argv[5]
    model = sys.argv[6]
    print('Starting Epoch: 0')
    print('Total Epoches:', epochs)
    training_and_val(epochs, base_dir, batch_size, num_channels, num_class, norm, loss_type, file_prefix, model)


if __name__ == "__main__":
   main()
