import torch
from torch import nn
import math
import torch.nn.functional as F

class BottleNeck(nn.Module):
    expansion=4

    def __init__(self,in_planes,planes,stride,downsample=None):
        super(BottleNeck,self).__init__()
        self.bottleneck=nn.Sequential(
            nn.Conv2d(in_planes,planes,kernel_size=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes,kernel_size=3,stride=stride,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes*self.expansion,kernel_size=1,bias=False),
            nn.BatchNorm2d(planes*self.expansion)
        )
        self.downsample=downsample
    
    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)
        x=self.bottleneck(x)
        return F.relu(x+identity)

class FPN(nn.Module):
    def __init__(self,bottleneck_num_list):
        super(FPN,self).__init__()
        self.bottleneck_num_list=bottleneck_num_list
        self.inplanes=64

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layer(planes=64,block_num=bottleneck_num_list[0],stride=1)
        self.layer2=self._make_layer(planes=128,block_num=bottleneck_num_list[1],stride=2)
        self.layer3=self._make_layer(planes=256,block_num=bottleneck_num_list[2],stride=2)
        self.layer4=self._make_layer(planes=512,block_num=bottleneck_num_list[3],stride=2)


    def _make_layer(self,planes,block_num,stride):
        downsample=None
        if stride!=1 or self.inplanes!=BottleNeck.expansion*planes:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,BottleNeck.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(BottleNeck.expansion*planes)
            )
        layers=[]
        layers.append(BottleNeck(self.inplanes,planes,stride,downsample=downsample))
        self.inplanes=planes*BottleNeck.expansion

        for i in range(1,block_num):
            layers.append(BottleNeck(self.inplanes,planes,stride=1,downsample=None))

        return nn.Sequential(*layers)
