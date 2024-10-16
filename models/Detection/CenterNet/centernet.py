from torch import nn
from torch import Tensor

import torch
import math
from models._init import fill_up_weights


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)   # gamma
            m.bias.data.fill_(0.0)      # beta
            m.running_mean.data.fill_(0.0)  # 初始化均值
            m.running_var.data.fill_(1.0)   # 初始化方差

class BackBoneDecoder(nn.Sequential):
    
    def __init__(self, in_channel, 
                 num_filters = [256, 128, 64], 
                 num_kernels = [4, 4, 4],
                 bn_momentum = 0.1
                 ):

        layers = []
        self.in_planes = in_channel
        for planes, kernel in zip(num_filters, num_kernels):
            convTrans = nn.ConvTranspose2d(
                self.in_planes,
                planes,
                kernel,
                2,
                1,
                bias=False
            )
            
            fill_up_weights(convTrans)

            layers.append(convTrans)
            
            layers.append(nn.BatchNorm2d(
                planes, momentum=bn_momentum
            ))
            
            layers.append(nn.ReLU(inplace=True))
            self.in_planes = planes

        super().__init__(*layers)
        init_weights(self)
    
class Head(nn.Module):

    def __init__(self, num_classes, channel = 64, bn_momentum = 0.1):
        super().__init__()

        # 热力图预测
        self.ht_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0)
        )
        
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
        )

        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
        )
        init_weights(self)

    def forward(self, x: Tensor):
        
        cls: Tensor = self.ht_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x) 

        return cls, wh, offset

class CenterNet(nn.Module):
    
    def __init__(self, backbone, in_channel, num_classes):
        
        super().__init__()
        self.backbone = backbone
        self.decoder = BackBoneDecoder(in_channel)
        self.head = Head(num_classes)

        # 初始化参数
        self.head.ht_head[-1].weight.data.fill_(0)
        self.head.ht_head[-1].bias.data.fill_(-2.19)

    
    def forward(self, x: Tensor):
        
        x = self.backbone(x)
        return self.head(self.decoder(x))
    




    
