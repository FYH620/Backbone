import torch
from torch import nn
import torch.nn.functional as F

class DetNetBottleneck(nn.Module):
    def __init__(self,in_planes,planes,stride=1,extra=False):
        super(DetNetBottleneck,self).__init__()
        self.bottleneck=nn.Sequential(
            nn.Conv2d(in_planes,planes,kernel_size=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=2,\
                dilation=2,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes,kernel_size=1,bias=False),
            nn.BatchNorm2d(planes)
        )
        self.extra=extra
        if self.extra:
            self.extra_conv=nn.Sequential(
                nn.Conv2d(in_planes,planes,kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self,x):
        identity=x
        out=self.bottleneck(x)
        if self.extra:
            identity=self.extra_conv(x)
        out+=identity
        return F.relu(out)

class ResNetBottleNeck(nn.Module):
    expansion=4

    def __init__(self,in_planes,planes,stride,downsample=None):
        super(ResNetBottleNeck,self).__init__()
        self.bottleneck=nn.Sequential(
            nn.Conv2d(in_planes,planes,kernel_size=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes*self.expansion,kernel_size=1,bias=False),
            nn.BatchNorm2d(planes*self.expansion)
        )
        self.downsample=downsample
    
    def forward(self,x):
        identity=x
        out=self.bottleneck(x)
        if self.downsample is not None:
            identity=self.downsample(x)
        out+=identity
        return F.relu(out)

class DetNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(DetNet,self).__init__()
        self.inplanes=64

        self.stage1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.stage2=self._make_layer(planes=64,block_num=3,stride=1)
        self.stage3=self._make_layer(planes=128,block_num=4,stride=2)
        self.stage4=self._make_layer(planes=256,block_num=6,stride=2)

        self.stage5=nn.Sequential(
            DetNetBottleneck(self.inplanes,256,stride=1,extra=True),
            DetNetBottleneck(256,256,stride=1,extra=False),
            DetNetBottleneck(256,256,stride=1,extra=False)
        )
        self.stage6=nn.Sequential(
            DetNetBottleneck(256,256,stride=1,extra=True),
            DetNetBottleneck(256,256,stride=1,extra=False),
            DetNetBottleneck(256,256,stride=1,extra=False)
        )

        self.smooth1=nn.Conv2d(self.inplanes,256,kernel_size=1,stride=1,padding=0)
        self.smooth2=nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)
        self.smooth3=nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)
        
        self.gap=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Linear(256,num_classes)

    def _make_layer(self,planes,block_num,stride):
        downsample=None
        if stride!=1 or self.inplanes!=ResNetBottleNeck.expansion*planes:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,ResNetBottleNeck.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(ResNetBottleNeck.expansion*planes)
            )
        layers=[]
        layers.append(ResNetBottleNeck(self.inplanes,planes,stride,downsample=downsample))
        self.inplanes=planes*ResNetBottleNeck.expansion

        for i in range(1,block_num):
            layers.append(ResNetBottleNeck(self.inplanes,planes,stride=1,downsample=None))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.stage4(x)
        p1=self.stage5(x)
        s1=self.smooth1(x)
        s2=self.smooth2(p1)
        out1=s1+s2
        p2=self.stage6(p1)
        s3=self.smooth3(p2)
        out2=s2+s3
        p2=self.gap(p2)
        p2=p2.view(p2.shape[0],-1)
        p2=self.fc(p2)
        return out1,out2,s3,p2

#detnet=DetNet(num_classes=1000)