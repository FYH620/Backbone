import torch
import torch.nn.functional as F
from torch import nn

class Block(nn.Module):
    
    def __init__(self,in_planes,out_planes,expansion,stride):
        super(Block,self).__init__()
        planes=in_planes*expansion
        self.stride=stride

        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,groups=planes,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv3=nn.Conv2d(planes,out_planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=nn.BatchNorm2d(out_planes)
        
        self.short=nn.Sequential()
        # use in_planes!=out_planes to control use conv1x1 only in the first num_block.
        if self.stride==1 and in_planes!=out_planes:
            self.short=nn.Sequential(
                nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_planes)
            )
    
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))
        if self.stride==1:
            out=out+self.short(x)
        return F.relu(out)


class MobileNetV2(nn.Module):
    def __init__(self,num_classes):
        super(MobileNetV2,self).__init__()
        # expansion out_planes num_blocks stride
        self.cfg=[
            (1,  16, 1, 1),
            (6,  24, 2, 1),
            (6,  32, 3, 2),
            (6,  64, 4, 2),
            (6,  96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1)
        ]
        self.conv1=nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(320,1280,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2=nn.BatchNorm2d(1280)
        self.layers=self._make_layers(in_planes=32)
        self.gap=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Linear(1280,num_classes)

    def _make_layers(self,in_planes):
        layers=[]
        for expansion,out_planes,num_blocks,stride in self.cfg:
            strides=[stride]+[1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes,out_planes,expansion,stride))
                in_planes=out_planes
        return nn.Sequential(*layers)

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.layers(x)
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.gap(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return F.softmax(x,dim=1)
        
# mobilenet=MobileNetV2(num_classes=1000)

