import torch
from torch import nn
import torch.nn.functional as F
from math import floor
from torchsummary import summary

class BottleNeck(nn.Module):
    def __init__(self,in_dims,growth_rate):
        super(BottleNeck,self).__init__()
        self.bn1=nn.BatchNorm2d(num_features=in_dims)
        self.conv1=nn.Conv2d(in_dims,4*growth_rate,kernel_size=1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(4*growth_rate)
        self.conv2=nn.Conv2d(4*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False)
    def forward(self,x):
        identity=x
        x=self.conv1(F.relu(self.bn1(x)))
        x=self.conv2(F.relu(self.bn2(x)))
        return torch.cat([x,identity],dim=1)

class DenseBlock(nn.Module):
    def __init__(self,in_dims,growth_rate,bottleneck_num):
        super(DenseBlock,self).__init__()
        layers=[]
        for i in range(bottleneck_num):
            layers.append(BottleNeck(in_dims=in_dims,growth_rate=growth_rate))
            in_dims+=growth_rate
        self.denseblock=nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.denseblock(x)
        return x

class Transition(nn.Module):
    def __init__(self,in_dims,out_dims):
        super(Transition,self).__init__()
        self.bn1=nn.BatchNorm2d(in_dims)
        self.conv1=nn.Conv2d(in_dims,out_dims,kernel_size=1,bias=False)
        self.avgpool=nn.AvgPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        x=self.conv1(F.relu(self.bn1(x)))
        x=self.avgpool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self,growth_rate,bottleneck_num_list,num_classes):
        super(DenseNet,self).__init__()

        out_dims=2*growth_rate
        self.conv1=nn.Conv2d(3,out_dims,kernel_size=7,stride=2,padding=3,bias=False)
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)

        self.denseblock1=DenseBlock(in_dims=out_dims,growth_rate=growth_rate,\
            bottleneck_num=bottleneck_num_list[0])
        
        out_dims+=(growth_rate*bottleneck_num_list[0])
        transition_out_dims=int(floor(out_dims*0.5))
        self.transition1=Transition(in_dims=out_dims,out_dims=transition_out_dims)
        out_dims=transition_out_dims

        self.denseblock2=DenseBlock(in_dims=out_dims,growth_rate=growth_rate,\
            bottleneck_num=bottleneck_num_list[1])
        
        out_dims+=(growth_rate*bottleneck_num_list[1])
        transition_out_dims=int(floor(out_dims*0.5))
        self.transition2=Transition(in_dims=out_dims,out_dims=transition_out_dims)
        out_dims=transition_out_dims

        self.denseblock3=DenseBlock(in_dims=out_dims,growth_rate=growth_rate,\
            bottleneck_num=bottleneck_num_list[2])
        
        out_dims+=(growth_rate*bottleneck_num_list[2])
        transition_out_dims=int(floor(out_dims*0.5))
        self.transition3=Transition(in_dims=out_dims,out_dims=transition_out_dims)
        out_dims=transition_out_dims

        self.denseblock4=DenseBlock(in_dims=out_dims,growth_rate=growth_rate,\
            bottleneck_num=bottleneck_num_list[3])
        
        out_dims+=(growth_rate*bottleneck_num_list[3])

        self.avgpool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Linear(in_features=out_dims,out_features=num_classes)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.denseblock1(x)
        x=self.transition1(x)
        x=self.denseblock2(x)
        x=self.transition2(x)
        x=self.denseblock3(x)
        x=self.transition3(x)
        x=self.denseblock4(x)
        x=self.avgpool(x)
        x=self.fc(x)
        return x

