import torch
import torch.nn.functional as F
from torch import nn

class Fire(nn.Module):
    def __init__(self,in_planes,squeeze_planes,expand_planes):
        super(Fire,self).__init__()
        self.conv1=nn.Conv2d(in_planes,squeeze_planes,kernel_size=1,stride=1,padding=0)
        self.bn1=nn.BatchNorm2d(squeeze_planes)
        self.conv2=nn.Conv2d(squeeze_planes,expand_planes,kernel_size=1,stride=1,padding=0)
        self.bn2=nn.BatchNorm2d(expand_planes)
        self.conv3=nn.Conv2d(squeeze_planes,expand_planes,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(expand_planes)

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        out1=self.bn2(self.conv2(x))
        out2=self.bn3(self.conv3(x))
        out=F.relu(torch.cat([out1,out2],dim=1))
        return out

