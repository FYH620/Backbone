import torch
from torch import nn
import torch.nn.functional as F
from math import floor

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
        self.growth_rate=growth_rate
        self.out_dims=2*self.growth_rate
        self.bottleneck_num_list=bottleneck_num_list
        self.num_classes=num_classes

        self.conv1=nn.Conv2d(3,self.out_dims,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.out_dims)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.denseblock1,self.transition1=self._make_denseblock_and_transition(0,with_transition=True)
        self.denseblock2,self.transition2=self._make_denseblock_and_transition(1,with_transition=True)
        self.denseblock3,self.transition3=self._make_denseblock_and_transition(2,with_transition=True)
        self.denseblock4=self._make_denseblock_and_transition(index=3,with_transition=False)

        self.bn2=nn.BatchNorm2d(self.out_dims)
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Linear(in_features=self.out_dims,out_features=self.num_classes)

    def _make_denseblock_and_transition(self,index,with_transition=True):

        denseblock=DenseBlock(in_dims=self.out_dims,growth_rate=self.growth_rate,\
            bottleneck_num=self.bottleneck_num_list[index])
        self.out_dims+=(self.growth_rate*self.bottleneck_num_list[index])

        if with_transition:
            transition_out_dims=int(floor(self.out_dims*0.5))
            transition=Transition(in_dims=self.out_dims,out_dims=transition_out_dims)
            self.out_dims=transition_out_dims
            return denseblock,transition

        return denseblock
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.maxpool(x)
        x=self.denseblock1(x)
        x=self.transition1(x)
        x=self.denseblock2(x)
        x=self.transition2(x)
        x=self.denseblock3(x)
        x=self.transition3(x)
        x=self.denseblock4(x)
        x=self.bn2(x)
        x=self.avgpool(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        x=F.softmax(x)
        return x

#densenet121=DenseNet(growth_rate=32,bottleneck_num_list=[6,12,24,16],num_classes=1000)
