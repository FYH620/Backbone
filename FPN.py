import torch
from torch import nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
    expansion=4

    def __init__(self,in_planes,planes,stride,downsample=None):
        super(BottleNeck,self).__init__()
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

class FPN(nn.Module):
    def __init__(self,bottleneck_num_list):
        super(FPN,self).__init__()
        self.bottleneck_num_list=bottleneck_num_list
        self.inplanes=64

        self.layer0=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.layer1=self._make_layer(planes=64,block_num=bottleneck_num_list[0],stride=1)
        self.layer2=self._make_layer(planes=128,block_num=bottleneck_num_list[1],stride=2)
        self.layer3=self._make_layer(planes=256,block_num=bottleneck_num_list[2],stride=2)
        self.layer4=self._make_layer(planes=512,block_num=bottleneck_num_list[3],stride=2)

        self.smooth1=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth2=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth3=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)

        self.lat_layer1=nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0)
        self.lat_layer2=nn.Conv2d(1024,256,kernel_size=1,stride=1,padding=0)
        self.lat_layer3=nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)
        self.lat_layer4=nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)

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

    def _upsample_and_add(self,small_feature_map,large_feature_map):
        _,_,height,width=large_feature_map.shape
        return F.upsample(input=small_feature_map,size=(height,width),mode='bilinear')+large_feature_map

    def forward(self,x):
        c1=self.layer0(x)
        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)

        p5=self.lat_layer1(c5)
        p4=self._upsample_and_add(small_feature_map=p5,large_feature_map=self.lat_layer2(c4))
        p3=self._upsample_and_add(small_feature_map=p4,large_feature_map=self.lat_layer3(c3))
        p2=self._upsample_and_add(small_feature_map=p3,large_feature_map=self.lat_layer4(c2))

        p4=self.smooth1(p4)
        p3=self.smooth2(p3)
        p2=self.smooth3(p2)

        return p2,p3,p4,p5

#fpn=FPN(bottleneck_num_list=[3,4,6,3])
