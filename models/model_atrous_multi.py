import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

'''
SegNet without deepest max pool and last 3 encoder convolution as
rate= 2 dilated convolution
Uses Leaky ReLU
Trained for multi losses using 3 different encoder units
Too big for our GPU
'''

class SegNet_atrous_multi(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet_atrous_multi, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        #self.dropout = nn.Dropout(0.3)

        self.conv53d_ce = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn53d_ce = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d_ce = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn52d_ce = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d_ce = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn51d_ce = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
#        self.convaspp = nn.Conv2d(2048, 512, kernel_size=1, dilation=1, padding=0)

        self.conv43d_ce = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d_ce = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d_ce = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d_ce = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d_ce = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d_ce = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d_ce = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d_ce = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d_ce = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d_ce = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d_ce = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d_ce = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d_ce = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d_ce = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d_ce = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d_ce = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv53d_dice = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn53d_dice = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d_dice = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn52d_dice = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d_dice = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn51d_dice = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

#        self.convaspp = nn.Conv2d(2048, 512, kernel_size=1, dilation=1, padding=0)

        self.conv43d_dice = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d_dice = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d_dice = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d_dice = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d_dice = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d_dice = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d_dice = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d_dice = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d_dice = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d_dice = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d_dice = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d_dice = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d_dice = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d_dice = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d_dice = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d_dice = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv53d_iou = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn53d_iou = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d_iou = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn52d_iou = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d_iou = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn51d_iou = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

#        self.convaspp = nn.Conv2d(2048, 512, kernel_size=1, dilation=1, padding=0)

        self.conv43d_iou = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d_iou = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d_iou = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d_iou = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d_iou = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d_iou = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d_iou = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d_iou = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d_iou = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d_iou = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d_iou = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d_iou = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d_iou = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d_iou = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d_iou = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d_iou = nn.BatchNorm2d(64, momentum= batchNorm_momentum)       
 
        self.conv12d_ce = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d_ce = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d_ce = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        
        self.conv12d_dice = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d_dice = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d_dice = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
   
        self.conv12d_iou = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d_iou = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d_iou = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

    def forward(self, x):
        # weight = 0.25
        # weight = numpy.array([weight])
        # weight = torch.from_numpy(weight)
        # Stage 1
        x11 = F.leaky_relu(self.bn11(self.conv11(x)), negative_slope=0.1)
        x12 = F.leaky_relu(self.bn12(self.conv12(x11)), negative_slope=0.1)
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x21 = F.leaky_relu(self.bn21(self.conv21(x1p)), negative_slope=0.1)
        x22 = F.leaky_relu(self.bn22(self.conv22(x21)), negative_slope=0.1)
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.leaky_relu(self.bn31(self.conv31(x2p)), negative_slope=0.1)
        x32 = F.leaky_relu(self.bn32(self.conv32(x31)), negative_slope=0.1)
        x33 = F.leaky_relu(self.bn33(self.conv33(x32)), negative_slope=0.1)
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x41 = F.leaky_relu(self.bn41(self.conv41(x3p)), negative_slope=0.1)
        x42 = F.leaky_relu(self.bn42(self.conv42(x41)), negative_slope=0.1)
        x43 = F.leaky_relu(self.bn43(self.conv43(x42)), negative_slope=0.1)
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

        # Stage 5
        x51 = F.leaky_relu(self.bn51(self.conv51(x4p)), negative_slope=0.1)
        x52 = F.leaky_relu(self.bn52(self.conv52(x51)), negative_slope=0.1)
        x53 = F.leaky_relu(self.bn53(self.conv53(x52)), negative_slope=0.1)
#        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=1,return_indices=True)
        #x5do = self.dropout(x53)
      
        # Stage 5d
#        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=1)
        x53d_ce = F.leaky_relu(self.bn53d_ce(self.conv53d_ce(x53)), negative_slope=0.1)
        x52d_ce = F.leaky_relu(self.bn52d_ce(self.conv52d_ce(x53d_ce)), negative_slope=0.1)
        x51d_ce = F.leaky_relu(self.bn51d_ce(self.conv51d_ce(x52d_ce)), negative_slope=0.1)
        
        x53d_dice = F.leaky_relu(self.bn53d_dice(self.conv53d_dice(x53)), negative_slope=0.1)
        x52d_dice = F.leaky_relu(self.bn52d_dice(self.conv52d_dice(x53d_dice)), negative_slope=0.1)
        x51d_dice = F.leaky_relu(self.bn51d_dice(self.conv51d_dice(x52d_dice)), negative_slope=0.1)
        
        x53d_iou = F.leaky_relu(self.bn53d_dice(self.conv53d_iou(x53)), negative_slope=0.1)
        x52d_iou = F.leaky_relu(self.bn52d_dice(self.conv52d_iou(x53d_iou)), negative_slope=0.1)
        x51d_iou = F.leaky_relu(self.bn51d_dice(self.conv51d_iou(x52d_iou)), negative_slope=0.1)
         #Stage aspp

#        x5cat = torch.cat((x5d, x53d, x52d, x51d), dim=1)
#        x5aspp = F.leaky_relu(self.convaspp(x5cat), negative_slope=0.1)

        # Stage 4d
        x4d_ce = F.max_unpool2d(x51d_ce, id4, kernel_size=2, stride=2)
        x43d_ce = F.leaky_relu(self.bn43d_ce(self.conv43d_ce(x4d_ce)), negative_slope=0.1)
        x42d_ce = F.leaky_relu(self.bn42d_ce(self.conv42d_ce(x43d_ce)), negative_slope=0.1)
        x41d_ce = F.leaky_relu(self.bn41d_ce(self.conv41d_ce(x42d_ce)), negative_slope=0.1)

        x4d_dice = F.max_unpool2d(x51d_dice, id4, kernel_size=2, stride=2)
        x43d_dice = F.leaky_relu(self.bn43d_dice(self.conv43d_dice(x4d_dice)), negative_slope=0.1)
        x42d_dice = F.leaky_relu(self.bn42d_dice(self.conv42d_dice(x43d_dice)), negative_slope=0.1)
        x41d_dice = F.leaky_relu(self.bn41d_dice(self.conv41d_dice(x42d_dice)), negative_slope=0.1)

        x4d_iou = F.max_unpool2d(x51d_iou, id4, kernel_size=2, stride=2)
        x43d_iou = F.leaky_relu(self.bn43d_iou(self.conv43d_iou(x4d_iou)), negative_slope=0.1)
        x42d_iou = F.leaky_relu(self.bn42d_iou(self.conv42d_iou(x43d_iou)), negative_slope=0.1)
        x41d_iou = F.leaky_relu(self.bn41d_iou(self.conv41d_iou(x42d_iou)), negative_slope=0.1)

        # Stage 3d
        x3d_ce = F.max_unpool2d(x41d_ce, id3, kernel_size=2, stride=2)
        x33d_ce = F.leaky_relu(self.bn33d_ce(self.conv33d_ce(x3d_ce)), negative_slope=0.1)
        x32d_ce = F.leaky_relu(self.bn32d_ce(self.conv32d_ce(x33d_ce)), negative_slope=0.1)
        x31d_ce = F.leaky_relu(self.bn31d_ce(self.conv31d_ce(x32d_ce)), negative_slope=0.1)

        x3d_dice = F.max_unpool2d(x41d_dice, id3, kernel_size=2, stride=2)
        x33d_dice = F.leaky_relu(self.bn33d_dice(self.conv33d_dice(x3d_dice)), negative_slope=0.1)
        x32d_dice = F.leaky_relu(self.bn32d_dice(self.conv32d_dice(x33d_dice)), negative_slope=0.1)
        x31d_dice = F.leaky_relu(self.bn31d_dice(self.conv31d_dice(x32d_dice)), negative_slope=0.1)

        x3d_iou = F.max_unpool2d(x41d_iou, id3, kernel_size=2, stride=2)
        x33d_iou = F.leaky_relu(self.bn33d_iou(self.conv33d_iou(x3d_iou)), negative_slope=0.1)
        x32d_iou = F.leaky_relu(self.bn32d_iou(self.conv32d_iou(x33d_iou)), negative_slope=0.1)
        x31d_iou = F.leaky_relu(self.bn31d_iou(self.conv31d_iou(x32d_iou)), negative_slope=0.1)

        # Stage 2d
        x2d_ce = F.max_unpool2d(x31d_ce, id2, kernel_size=2, stride=2)
        x22d_ce = F.leaky_relu(self.bn22d_ce(self.conv22d_ce(x2d_ce)), negative_slope=0.1)
        x21d_ce = F.leaky_relu(self.bn21d_ce(self.conv21d_ce(x22d_ce)), negative_slope=0.1)

        x2d_dice = F.max_unpool2d(x31d_dice, id2, kernel_size=2, stride=2)
        x22d_dice = F.leaky_relu(self.bn22d_dice(self.conv22d_dice(x2d_dice)), negative_slope=0.1)
        x21d_dice = F.leaky_relu(self.bn21d_dice(self.conv21d_dice(x22d_dice)), negative_slope=0.1)

        x2d_iou = F.max_unpool2d(x31d_iou, id2, kernel_size=2, stride=2)
        x22d_iou = F.leaky_relu(self.bn22d_iou(self.conv22d_iou(x2d_iou)), negative_slope=0.1)
        x21d_iou = F.leaky_relu(self.bn21d_iou(self.conv21d_iou(x22d_iou)), negative_slope=0.1)
        # Stage 1d
        x1d_ce = F.max_unpool2d(x21d_ce, id1, kernel_size=2, stride=2)
        x1d_dice = F.max_unpool2d(x21d_dice, id1, kernel_size=2, stride=2)
        x1d_iou = F.max_unpool2d(x21d_iou, id1, kernel_size=2, stride=2)
        x12d_dice = F.leaky_relu(self.bn12d_dice(self.conv12d_dice(x1d_dice)), negative_slope=0.1)
        x11d_dice = self.conv11d_dice(x12d_dice)
        x12d_ce = F.leaky_relu(self.bn12d_ce(self.conv12d_ce(x1d_ce)), negative_slope=0.1)
        x11d_ce = self.conv11d_ce(x12d_ce)
        x12d_iou = F.leaky_relu(self.bn12d_iou(self.conv12d_iou(x1d_iou)), negative_slope=0.1)
        x11d_iou = self.conv11d_iou(x12d_iou)


        return x11d_ce, x11d_dice, x11d_iou

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)
