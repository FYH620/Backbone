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
