import argparse
import os
import scipy.misc
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from load_road import make_data_splits
from load_road_1c import make_data_splits
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

def test(base_dir, batch_size):
        # Define Dataloader
    train_loader, val_loader, test_loader, nclass = make_data_splits(base_dir, batch_size=4)

        # Define network
    with open(os.path.join(os.path.join(base_dir, 'test.txt')), "r") as f:
            lines = f.read().splitlines()    

    model = SegNet_atrous(1,2)

    model = model.cuda()
    model.load_state_dict(load("/home/ahana/pytorch_road/models/SegNet_pre_atrous_1c_n_ce_lr_001_75"))
    model.eval()
    tbar = tqdm(test_loader)
    overall_confusion_matrix = None
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()
        with no_grad():
            output = model(image)
        #pred = output.data.cpu().numpy()
        #target = target.cpu().numpy()
        #pred = np.argmax(pred, axis=1)
        #for j in range(0,4):
            #outFilepath = "/home/ahana/road_data/pred_train_4c/"+lines[i*batch_size+j]+".png"
            #outFilepath_target = "/home/ahana/road_data/target_test_norm_atrous/"+str(i)+"_"+str(j)+".png"
            #to_save = pred[j,:,:]
            #target_to_save = target[j,:,:]
            #scipy.misc.toimage(to_save).save(outFilepath)
            #scipy.misc.toimage(target_to_save).save(outFilepath_target)

            # Add batch sample into evaluator
        #target_f = target.flatten()
        #pred_f = pred.flatten()
            # print(np.count_nonzero(target_f))
            # print(np.count_nonzero(pred_f))
        #current_confusion_matrix = confusion_matrix(y_true=target_f, y_pred=pred_f, labels=[0, 1])
        
        #if overall_confusion_matrix is not None:
        #    overall_confusion_matrix += current_confusion_matrix
        #else:
        #    overall_confusion_matrix = current_confusion_matrix
    
        
    #intersection = overall_confusion_matrix[1][1]
    #ground_truth_set = overall_confusion_matrix.sum(axis=1)
    #predicted_set = overall_confusion_matrix.sum(axis=0)
    #union =  overall_confusion_matrix[0][1] + overall_confusion_matrix[1][0]

    #intersection_over_union = intersection / union.astype(np.float32)
    #RMIoU = intersection/(overall_confusion_matrix[0][1] + overall_confusion_matrix[1][0] + intersection)
    #RMIoU = intersection/(ground_truth_set + predicted_set - intersection)   
        # Fast test during the training
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
            #if epoch == epochs-1:
                #for j in range(0,4):
                    #outFilepath = "/home/ahana/road_data/results_norm_deep/"+str(i)+"_"+str(j)+".png"
                    #outFilepath_target = "/home/ahana/road_data/target_norm_deep/"+str(i)+"_"+str(j)+".png"
                    #to_save = pred[j,:,:]
                    #target_to_save = target[j,:,:]
                    #scipy.misc.toimage(to_save).save(outFilepath)
                    #scipy.misc.toimage(target_to_save).save(outFilepath_target)

            #if epoch == epochs-51:
                #for j in range(0,4):
                    #outFilepath = "/home/ahana/road_data/results_pre_norm_deep/"+str(i)+"_"+str(j)+".png"
                    #outFilepath_target = "/home/ahana/road_data/target_pre_norm_deep/"+str(i)+"_"+str(j)+".png"
                    #to_save = pred[j,:,:]
                    #target_to_save = target[j,:,:]
                    #scipy.misc.toimage(to_save).save(outFilepath)
                    #scipy.misc.toimage(target_to_save).save(outFilepath_target)
            # Add batch sample into evaluator
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
