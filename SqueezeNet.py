import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary

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

class SqueezeNet(nn.Module):
    def __init__(self,num_classes):
        super(SqueezeNet,self).__init__()
        self.conv1=nn.Conv2d(3,96,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(96)
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.fire2=Fire(in_planes=96,squeeze_planes=16,expand_planes=64)
        self.fire3=Fire(in_planes=128,squeeze_planes=16,expand_planes=64)
        self.fire4=Fire(in_planes=128,squeeze_planes=32,expand_planes=128)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.fire5=Fire(in_planes=256,squeeze_planes=32,expand_planes=128)
        self.fire6=Fire(in_planes=256,squeeze_planes=48,expand_planes=192)
        self.fire7=Fire(in_planes=384,squeeze_planes=48,expand_planes=192)
        self.fire8=Fire(in_planes=384,squeeze_planes=64,expand_planes=256)
        self.maxpool3=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.fire9=Fire(in_planes=512,squeeze_planes=64,expand_planes=256)
        self.dropout=nn.Dropout(p=0.5)
        self.conv10=nn.Conv2d(512,num_classes,kernel_size=1,stride=1,padding=0)
        self.bn10=nn.BatchNorm2d(num_classes)
        self.gap=nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.maxpool1(x)
        x=self.fire2(x)
        x=self.fire3(x)
        x=self.fire4(x)
        x=self.maxpool2(x)
        x=self.fire5(x)
        x=self.fire6(x)
        x=self.fire7(x)
        x=self.fire8(x)
        x=self.maxpool3(x)
        x=self.fire9(x)
        x=self.dropout(x)
        x=F.relu(self.bn10(self.conv10(x)))
        x=self.gap(x)
        x=x.view(x.size(0),-1)
        return F.softmax(x,dim=1)

squeezenet=SqueezeNet(num_classes=1000)
print(summary(squeezenet,input_size=(3,224,224)))