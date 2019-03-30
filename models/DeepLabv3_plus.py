import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class SeperableConv2D(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeperableConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_dims, in_dims, kernel_size, 
                                stride, padding, dilation, groups=in_dims, bias=bias)
        self.pointwise = nn.Conv2d(in_dims, out_dims, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=bias)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
    
def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size-1) * (rate-1)
    pad_total = kernel_size_effective-1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end), mode='reflect')
    return padded_inputs
        
        

class SeperableConv2D_same(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeperableConv2D_same, self).__init__()
        self.conv1 = nn.Conv2d(in_dims, in_dims, kernel_size, stride, 0, dilation, groups=in_dims, bias=bias)
        self.pointwise = nn.Conv2d(in_dims, out_dims, 1, 1, 0, 1, 1, bias=bias)
        
    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
    
class Block(nn.Module):
    def __init__(self, in_dims, out_dims, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_dims != in_dims or stride != 1:
            self.skip = nn.Conv2d(in_dims, out_dims, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_dims)
        else:
            self.skip = None
            
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_dims
        if grow_first:
            rep.append(self.relu)
            rep.append(SeperableConv2D_same(in_dims, out_dims, 3, stride=1, dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(out_dims))
            filters = out_dims
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeperableConv2D_same(filters, filters, 3, stride=1, dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeperableConv2D_same(in_dims, out_dims, 3, stride=1, dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(out_dims))
        if not start_with_relu:
            rep = rep[1:]
        if stride != 1:
            rep.append(SeperableConv2D_same(out_dims, out_dims, 3, stride=2))
        self.rep = nn.Sequential(*rep)
        
    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x
        out += skip
        return out

class Xception(nn.Module):
    def __init__(self, in_dims=3, pretrained=False):
        super(Xception, self).__init__()
        # entry flow
        self.conv1 = nn.Conv2d(in_dims, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=2, start_with_relu=True, grow_first=True)
        # middle blocks
        self.block4 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
        self.block20 = Block(728, 1024, reps=2, dilation=2, start_with_relu=True, grow_first=False)
        self.conv3 = SeperableConv2D_same(1024, 1536, 3, stride=1, dilation=2)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeperableConv2D_same(1536, 1536, 3, stride=1, dilation=2)
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeperableConv2D_same(1536, 2048, 3, stride=1, dilation=2)
        self.bn5 = nn.BatchNorm2d(2048)
        self._init_weights()
        if pretrained:
            self._load_xception_weights()
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_feat
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _load_xception_weights(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
                
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ASPP_Module(nn.Module):
    def __init__(self, in_dims, out_dims, rate, padding='fixed'):
        super(ASPP_Module, self).__init__()
        #self.atrous_conv = nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate, dilation=rate)
        self.conv1 = nn.Conv2d(in_dims, in_dims, 3, stride=1, padding=1 if padding=='same' else 0, 
                                   dilation=rate, groups=in_dims, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dims)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_dims)
        self.pointwise = nn.Conv2d(in_dims, out_dims, 1, 1, 0, 1, 1, bias=False)
        self.padding = padding
        self._init_weights()
        
    def forward(self, x):
        if self.padding == 'fixed':
            x = fixed_padding(x, self.conv1.kernel_size[0], self.conv1.dilation[0])
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=21, pretrained=False):
        print(f'Constructing Deeplabv3+ with {in_channels} input channels and {num_classes} classes')
        super(DeepLabv3_plus, self).__init__()
        self.xception_features = Xception(in_dims=in_channels, pretrained=pretrained)
        
        rates = [1, 6, 12, 18]
        self.aspp1 = ASPP_Module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_Module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_Module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_Module(2048, 256, rate=rates[3])
        self.relu = nn.ReLU(inplace=True)
        
        #self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(2048, 256, 1, stride=1))
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                             nn.Conv2d(2048, 256, 1, stride=1), 
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))
        
        self.conv1 = nn.Conv2d(1280, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.drop = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(128, 48, 1)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.last_conv = nn.Sequential(ASPP_Module(304, 256, 1, padding='same'), 
                                       ASPP_Module(256, 256, 1, padding='same'), 
                                       nn.Conv2d(256, num_classes, 1, stride=1))
        
        #self.last_conv = nn.Sequential(nn.Conv2d(304, 256, 3, stride=1, padding=1),
        #                               nn.BatchNorm2d(256),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(256, 256, 3, stride=1, padding=1),
        #                               nn.BatchNorm2d(256),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(256, num_classes, 1, stride=1))
        
    def forward(self, x):
        x, low_level_feat = self.xception_features(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        x = self.drop(self.relu(self.bn1(self.conv1(x))))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        low_level_feat = self.relu(self.bn2(self.conv2(low_level_feat)))
        
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

