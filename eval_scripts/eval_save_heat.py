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
#from load_road.load_road_pred import make_data_splits_p
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
from metrics.iou import IoU
import numpy

'''
A code to execute test for a given model and save heat maps:
    Args: num_channels, num_classes, model_name
          num_channels: number of input channels, also used to indicate augmented data.
                        0 for the model that uses 3 predictions
                        5 for 3 channel augmented
                        2 for 1 channel augmented
                        9 for 8 channel augmented
          num_classes: how many classes to be predicted
          num_classes: how many classes to be predicted
          cat_dir: directory that has the labels
          norm: whether to use normalize or not. 0 or 1.
          model_name: name of trained model to load
          split: train, val, test
          save_dir: for saving the heeatmaps
          model: model type: whether it uses GN or BN, uses dropout or not, and for how many channels
                 refer to models directory for morew info on different models
'''

def test(base_dir, batch_size, num_channels, num_class, cat_dir, norm, model_name, split, save_dir, model):
    # Define Dataloader
    print(split)
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
    if num_channels == 8:
        train_loader, val_loader, test_loader, nclass = make_data_splits_hs(base_dir, num_class, cat_dir, norm, 'eval', batch_size=4)
    if num_channels == 0: # for using with the 4 predictions
        train_loader, val_loader, test_loader, nclass = make_data_splits_p(base_dir, batch_size=4)
        num_channels = 4
    # List of file names
    if split == 'train':
        with open(os.path.join(os.path.join(base_dir, 'train.txt')), "r") as f:
            lines = f.read().splitlines()    
    if split == 'val':
        with open(os.path.join(os.path.join(base_dir, 'valid.txt')), "r") as f:
            lines = f.read().splitlines()
    if split == 'test':
        with open(os.path.join(os.path.join(base_dir, 'test.txt')), "r") as f:
            lines = f.read().splitlines()
    # Define and load network
    if model == 'hs': #hyperspectral with dropout
        model = SegNet_atrous_hs(num_channels, num_class)
    elif model == 'shallow':
        model = SegNet_shallow(num_channels, num_class)
    elif model == 'GN':
        model = SegNet_atrous_GN(num_channels, num_class)
    elif model == 'GN_dropout':
        model = SegNet_atrous_GN_dropout(num_channels, num_class)
    else: # no GN or dropout
        model = SegNet_atrous(num_channels, num_class)

    model = model.cuda()
    model.load_state_dict(load("/home/ahana/pytorch_road/trained_models/"+model_name))
    # Start tests
    model.eval()
    metric = IoU(num_class)
    if split == 'train':
        tbar = tqdm(train_loader)
    if split == 'val':
        tbar = tqdm(val_loader)
    if split == 'test':
        tbar = tqdm(test_loader)
    overall_confusion_matrix = None
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()
        with no_grad():
            output = model(image)
        
        pred = output.data.cpu()
        target = target.cpu()
        metric.add(pred, target)

        # Code to use for saving output heat maps
        fn = nn.Softmax()
        output = fn(output)
        pred_save = output.data.cpu().numpy()
        target_save = target.cpu().numpy()
        #pred_save = np.argmax(pred, axis=1)
        for j in range(0,4):
            outFilepath = "/home/ahana/road_data/"+save_dir+"/"+lines[i*batch_size+j]+".png"
            #outFilepath_target = "/home/ahana/road_data/target_test_norm_atrous/"+str(i)+"_"+str(j)+".png"
            to_save = pred_save[j,1,:,:]
            to_save = to_save * 200.0
            padded_save = to_save.astype(dtype=np.uint8)
            unpad_save = padded_save[6:506,6:506]
            #to_save = to_save 
            #target_to_save = target[j,:,:]
            scipy.misc.toimage(unpad_save, cmin=0,cmax=255).save(outFilepath)
            #scipy.misc.toimage(target_to_save).save(outFilepath_target)

    iou, miou = metric.value()
    print('Test:')
    print("IoU: {}, MIoU: {}".format(iou, miou))
    metric.reset()

def main():

    base_dir = "/home/ahana/road_data"
    batch_size = 4
    num_channels = int(sys.argv[1])
    num_class = int(sys.argv[2])
    cat_dir = sys.argv[3]
    norm = int(sys.argv[4])
    model_name = sys.argv[5]
    split = sys.argv[6]
    save_dir = sys.argv[7]
    model = sys.argv[8]
    test(base_dir, batch_size, num_channels, num_class, cat_dir, norm, model_name, split, save_dir, model)


if __name__ == "__main__":
   main()
