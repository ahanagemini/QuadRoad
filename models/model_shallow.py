import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

'''
SegNet without deepest max pool and last 3 encoder convolution as 
rate= 2 dilated convolution
Uses Leaky ReLU
'''

class SegNet_shallow(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet_shallow, self).__init__()

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

        self.dropout = nn.Dropout(0.3)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)


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

        x31 = F.leaky_relu(self.bn31(self.conv31(x2p)), negative_slope=0.1)
        x32 = F.leaky_relu(self.bn32(self.conv32(x31)), negative_slope=0.1)
        x33 = F.leaky_relu(self.bn33(self.conv33(x32)), negative_slope=0.1)
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        x3d = F.max_unpool2d(x3p, id3, kernel_size=2, stride=2)
        x33d = F.leaky_relu(self.bn33d(self.conv33d(x3d)), negative_slope=0.1)
        x32d = F.leaky_relu(self.bn32d(self.conv32d(x33d)), negative_slope=0.1)
        x31d = F.leaky_relu(self.bn31d(self.conv31d(x32d)), negative_slope=0.1)
        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.leaky_relu(self.bn22d(self.conv22d(x2d)), negative_slope=0.1)
        x21d = F.leaky_relu(self.bn21d(self.conv21d(x22d)), negative_slope=0.1)

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.leaky_relu(self.bn12d(self.conv12d(x1d)), negative_slope=0.1)
        x11d = self.conv11d(x12d)

        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)
