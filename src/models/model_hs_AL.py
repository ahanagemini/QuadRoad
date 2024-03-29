import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

'''
Model for training on hyperspectral data
Uses Leaky Relu and dilated conv
Modify SegNet for hyperspectral by replacing 
    max-pool and unpool with upsampling in some cases

'''

class SegNet_hs_AL(nn.Module):
    """
    Class to implement segnet for hyper-spectral images 
    Uses atrous convolutions, leaky relu and dropout(optional)
    """
    def __init__(self,input_nbr,label_nbr, dropout):
        """
        :param input_nbr: number of input channels
        :param label_nbr: number of output channels
        :param dropout: whether to use dropout or not
        """
        super(SegNet_hs_AL, self).__init__()
        self.use_dropout = dropout
        batchNorm_momentum = 0.1
        # Encoder operations
        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.up1 = nn.UpsamplingBilinear2d(size=(256,256))        

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
        
        self.dropout = nn.Dropout(0.3)
        # Decoder operations
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

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

        self.up2 = nn.UpsamplingBilinear2d(size=(512, 512))
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)


    def forward(self, x):
        """
        Function to apply the model on the input x
        Args:
            x: input data
        Returns:
            x11d: The output of the model
        """
        # Stage 1
        x11 = F.leaky_relu(self.bn11(self.conv11(x)), negative_slope=0.1)
        x12 = F.leaky_relu(self.bn12(self.conv12(x11)), negative_slope=0.1)
        x1p = self.up1(x12)

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
        
        if self.use_dropout:
            x5do = self.dropout(x53)
      
            # Stage 5d
            x53d = F.leaky_relu(self.bn53d(self.conv53d(x5do)), negative_slope=0.1)
            x52d = F.leaky_relu(self.bn52d(self.conv52d(x53d)), negative_slope=0.1)
            x51d = F.leaky_relu(self.bn51d(self.conv51d(x52d)), negative_slope=0.1)

        else:
            x53d = F.leaky_relu(self.bn53d(self.conv53d(x53)), negative_slope=0.1)
            x52d = F.leaky_relu(self.bn52d(self.conv52d(x53d)), negative_slope=0.1)
            x51d = F.leaky_relu(self.bn51d(self.conv51d(x52d)), negative_slope=0.1)

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.leaky_relu(self.bn43d(self.conv43d(x4d)), negative_slope=0.1)
        x42d = F.leaky_relu(self.bn42d(self.conv42d(x43d)), negative_slope=0.1)
        x41d = F.leaky_relu(self.bn41d(self.conv41d(x42d)), negative_slope=0.1)

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.leaky_relu(self.bn33d(self.conv33d(x3d)), negative_slope=0.1)
        x32d = F.leaky_relu(self.bn32d(self.conv32d(x33d)), negative_slope=0.1)
        x31d = F.leaky_relu(self.bn31d(self.conv31d(x32d)), negative_slope=0.1)

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.leaky_relu(self.bn22d(self.conv22d(x2d)), negative_slope=0.1)
        x21d = F.leaky_relu(self.bn21d(self.conv21d(x22d)), negative_slope=0.1)

        # Stage 1d
        x1d = self.up2(x21d)
        x12d = F.leaky_relu(self.bn12d(self.conv12d(x1d)), negative_slope=0.1)
        x11d = self.conv11d(x12d)

        return x11d

    def load_from_segnet(self, model_path):
        """
        Function to load a saved model
        Args:
            model_path: Path to load model from
        """
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)
