import argparse
import os
import scipy.misc
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from load_road_3c import make_data_splits
from torchvision import models
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
from torchsummary import summary
from model import SegNet
from model_leaky import SegNet_leaky
from torch import no_grad
from torch import FloatTensor
from torch import save
import matplotlib.pyplot as plt
from model_deep import SegNet_deep
from model_atrous import SegNet_atrous
from DeepLabv3_plus import DeepLabv3_plus

def training_and_val(epochs, base_dir, batch_size):
        # Define Dataloader
    train_loader, val_loader, test_loader, nclass = make_data_splits(base_dir, batch_size=4)

        # Define network
    
    model = SegNet(3,2)
    # model = DeepLabv3_plus(in_channels=3, num_classes=2)
#    model.fc = nn.Linear(num_ftrs, nclass)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
            # self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    weights = [0.29, 1.69]
    class_weights = FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = criterion.cuda()
        
        # Define lr scheduler
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            # args.epochs, len(self.train_loader))

        # Using cuda
    model = model.cuda()
    best = 0.0
    losses = []
    ious = []
    best_epoch = 0
    for epoch in range(0, epochs):

        train_loss = 0.0
        model.train()
        tbar = tqdm(train_loader)
        num_img_tr = len(train_loader)
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
        print('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        losses.append(train_loss)
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
            # print(np.count_nonzero(target_f))
            # print(np.count_nonzero(pred_f))
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
        if RMIoU[1] > best:
            best = RMIoU[1]
            best_epoch = epoch
            #save(model.state_dict(), "/home/ahana/pytorch_road/model_best/SegNet_best_5em0")
            save(model.state_dict(), "/home/ahana/pytorch_road/model_best/SegNet_best_lr_001")
        if epoch % 10 == 0:
            #save(model.state_dict(), "/home/ahana/pytorch_road/model_best/SegNet_pre_atrous_5em0")
            outfile = "/home/ahana/pytorch_road/model_best/SegNet_pre_lr_001_" + str(epoch)
            save(model.state_dict(), outfile)

        if epoch == 75:
            outfile = "/home/ahana/pytorch_road/model_best/SegNet_pre_lr_001_75"
            save(model.state_dict(), outfile)

        #save(model.state_dict(), "/home/ahana/pytorch_road/model_best/SegNet_final_atrous_5em0")
        save(model.state_dict(), "/home/ahana/pytorch_road/model_best/SegNet_final_lr_001")
        ious.append(RMIoU[1])
        # Fast test during the training
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
        print("RMIoU: {}, Intersection: {}, Ground truth: {}, Predicted: {}, Best: {}".format(RMIoU, intersection, ground_truth_set, predicted_set, best))

    
    plt.plot(losses)
    plt.savefig("/home/ahana/pytorch_road/loss_SegNet_lr_001.png")
    plt.clf()
    plt.plot(ious)
    print(losses)
    print(ious)
    plt.savefig("/home/ahana/pytorch_road/iou_SegNet_lr_001.png")    
    losses = [(i,x) for i,x in enumerate(losses)]
    ious = [(i,x) for i,x in enumerate(ious)]
    fp = open("SegNet_lr_001.txt", "w")
    fp.write(str(losses))
    fp.write("\n")
    fp.write(str(ious))
    fp.write("\n")
    fp.write(str(best_epoch))
    fp.close()

def main():

    base_dir = "/home/ahana/road_data"
    epochs = 80
    batch_size = 4
    print('Starting Epoch: 0')
    print('Total Epoches:', epochs)
    training_and_val(epochs, base_dir, batch_size)


if __name__ == "__main__":
   main()
