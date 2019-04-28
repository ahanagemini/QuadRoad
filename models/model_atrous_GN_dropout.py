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

class SegNet_atrous_GN_dropout(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet_atrous_GN_dropout, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.gn11 = nn.GroupNorm(32, 64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.gn12 = nn.GroupNorm(32, 64)
        self.do1 = nn.Dropout(0.5)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn21 = nn.GroupNorm(32, 128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.gn22 = nn.GroupNorm(32, 128)
        self.do2 = nn.Dropout(0.5)


        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.gn31 = nn.GroupNorm(32, 256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.gn32 = nn.GroupNorm(32, 256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.gn33 = nn.GroupNorm(32, 256)
        self.do3 = nn.Dropout(0.5)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.gn41 = nn.GroupNorm(32, 512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.gn42 = nn.GroupNorm(32, 512)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.gn43 = nn.GroupNorm(32, 512)
        self.do4 = nn.Dropout(0.5)


        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.gn51 = nn.GroupNorm(32, 512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.gn52 = nn.GroupNorm(32, 512)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.gn53 = nn.GroupNorm(32, 512)
        self.do5 = nn.Dropout(0.5)
        #self.dropout = nn.Dropout(0.3)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.gn53d = nn.GroupNorm(32, 512)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.gn52d = nn.GroupNorm(32, 512)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.gn51d = nn.GroupNorm(32, 512)
        self.dod5 = nn.Dropout(0.5)
#        self.convaspp = nn.Conv2d(2048, 512, kernel_size=1, dilation=1, padding=0)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.gn43d = nn.GroupNorm(32, 512)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.gn42d = nn.GroupNorm(32, 512)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.gn41d = nn.GroupNorm(32, 256)
        self.dod4 = nn.Dropout(0.5)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.gn33d = nn.GroupNorm(32, 256)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.gn32d = nn.GroupNorm(32, 256)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.gn31d = nn.GroupNorm(32, 128)
        self.dod3 = nn.Dropout(0.5)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.gn22d = nn.GroupNorm(32, 128)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.gn21d = nn.GroupNorm(32, 64)
        self.dod2 = nn.Dropout(0.5)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.gn12d = nn.GroupNorm(32, 64)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)


    def forward(self, x):
        # weight = 0.25
        # weight = numpy.array([weight])
        # weight = torch.from_numpy(weight)
        # Stage 1
        x11 = F.leaky_relu(self.gn11(self.conv11(x)), negative_slope=0.1)
        x12 = F.leaky_relu(self.gn12(self.conv12(x11)), negative_slope=0.1)
        x12do = self.do1(x12)
        x1p, id1 = F.max_pool2d(x12do,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x21 = F.leaky_relu(self.gn21(self.conv21(x1p)), negative_slope=0.1)
        x22 = F.leaky_relu(self.gn22(self.conv22(x21)), negative_slope=0.1)
        x22do = self.do2(x22)
        x2p, id2 = F.max_pool2d(x22do,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.leaky_relu(self.gn31(self.conv31(x2p)), negative_slope=0.1)
        x32 = F.leaky_relu(self.gn32(self.conv32(x31)), negative_slope=0.1)
        x33 = F.leaky_relu(self.gn33(self.conv33(x32)), negative_slope=0.1)
        x33do = self.do3(x33)
        x3p, id3 = F.max_pool2d(x33do,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x41 = F.leaky_relu(self.gn41(self.conv41(x3p)), negative_slope=0.1)
        x42 = F.leaky_relu(self.gn42(self.conv42(x41)), negative_slope=0.1)
        x43 = F.leaky_relu(self.gn43(self.conv43(x42)), negative_slope=0.1)
        x43do = self.do4(x43)
        x4p, id4 = F.max_pool2d(x43do,kernel_size=2, stride=2,return_indices=True)

        # Stage 5
        x51 = F.leaky_relu(self.gn51(self.conv51(x4p)), negative_slope=0.1)
        x52 = F.leaky_relu(self.gn52(self.conv52(x51)), negative_slope=0.1)
        x53 = F.leaky_relu(self.gn53(self.conv53(x52)), negative_slope=0.1)
#        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=1,return_indices=True)
        x53do = self.do5(x53)
      
        # Stage 5d
#        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=1)
        x53d = F.leaky_relu(self.gn53d(self.conv53d(x53do)), negative_slope=0.1)
        x52d = F.leaky_relu(self.gn52d(self.conv52d(x53d)), negative_slope=0.1)
        x51d = F.leaky_relu(self.gn51d(self.conv51d(x52d)), negative_slope=0.1)
        x51ddo = self.dod5(x51d)
        #Stage aspp

#        x5cat = torch.cat((x5d, x53d, x52d, x51d), dim=1)
#        x5aspp = F.leaky_relu(self.convaspp(x5cat), negative_slope=0.1)

        # Stage 4d
        x4d = F.max_unpool2d(x51ddo, id4, kernel_size=2, stride=2)
        x43d = F.leaky_relu(self.gn43d(self.conv43d(x4d)), negative_slope=0.1)
        x42d = F.leaky_relu(self.gn42d(self.conv42d(x43d)), negative_slope=0.1)
        x41d = F.leaky_relu(self.gn41d(self.conv41d(x42d)), negative_slope=0.1)
        x41ddo = self.dod4(x41d)
        # Stage 3d
        x3d = F.max_unpool2d(x41ddo, id3, kernel_size=2, stride=2)
        x33d = F.leaky_relu(self.gn33d(self.conv33d(x3d)), negative_slope=0.1)
        x32d = F.leaky_relu(self.gn32d(self.conv32d(x33d)), negative_slope=0.1)
        x31d = F.leaky_relu(self.gn31d(self.conv31d(x32d)), negative_slope=0.1)
        x31ddo = self.dod3(x31d)
        # Stage 2d
        x2d = F.max_unpool2d(x31ddo, id2, kernel_size=2, stride=2)
        x22d = F.leaky_relu(self.gn22d(self.conv22d(x2d)), negative_slope=0.1)
        x21d = F.leaky_relu(self.gn21d(self.conv21d(x22d)), negative_slope=0.1)
        x21ddo = self.dod2(x21d)
        # Stage 1d
        x1d = F.max_unpool2d(x21ddo, id1, kernel_size=2, stride=2)
        x12d = F.leaky_relu(self.gn12d(self.conv12d(x1d)), negative_slope=0.1)
        x11d = self.conv11d(x12d)

        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)
