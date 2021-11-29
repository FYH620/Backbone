import torch
import torch.nn as nn

def conv3x3(in_channels,out_channels,stride):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,\
        padding=1,bias=False)

def conv1x1(in_channels,out_channels,stride):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,\
        stride=stride,bias=False)

class ConvBottleneck(nn.Module):

    def __init__(self,in_channels,channels_expansion,stride_expansion,first_construct):

        super(ConvBottleneck,self).__init__()
        self.in_channels=in_channels
        self.channels_expansion=channels_expansion
        self.stride_expansion=stride_expansion

        out_channels = in_channels if first_construct else int(in_channels/2)

        self.conv1=conv1x1(in_channels,out_channels,stride=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu1=nn.ReLU()
        self.conv2=conv3x3(out_channels,out_channels,stride=1*stride_expansion)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu2=nn.ReLU()
        self.conv3=conv1x1(out_channels,in_channels*channels_expansion,stride=1)
        self.bn3=nn.BatchNorm2d(in_channels*channels_expansion)
        self.relu3=nn.ReLU()

        self.conv4=conv1x1(in_channels,in_channels*channels_expansion,stride=1*stride_expansion)
        self.bn4=nn.BatchNorm2d(in_channels*channels_expansion)
    
    def forward(self,x):
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu1(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu2(out)

        out=self.conv3(out)
        out=self.bn3(out)
        
        x=self.conv4(x)
        x=self.bn4(x)

        out+=x
        out=self.relu3(out)

        return out

class IdentityBottleneck(nn.Module):
    def __init__(self,in_channels):

        super(IdentityBottleneck,self).__init__()
        self.in_channels=in_channels
        out_channels=int(in_channels/4)

        self.conv1=conv1x1(in_channels,out_channels,stride=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu1=nn.ReLU()
        self.conv2=conv3x3(out_channels,out_channels,stride=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu2=nn.ReLU()
        self.conv3=conv1x1(out_channels,in_channels,stride=1)
        self.bn3=nn.BatchNorm2d(in_channels)
        self.relu3=nn.ReLU()
    
    def forward(self,x):
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu1(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu2(out)

        out=self.conv3(out)
        out=self.bn3(out)
        
        out+=x
        out=self.relu3(out)

        return out

class ResNet(nn.Module):

    def __init__(self,conv_bottleneck,identity_bottleneck,num_classes=1000):

        self.conv_bottleneck=conv_bottleneck
        self.identity_bottleneck=identity_bottleneck
        super(ResNet,self).__init__()

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.stage1=self._make_stage(layer_num=3,in_channels=64,channels_expansion=4,stride_expansion=1,first_construct=True)
        self.stage2=self._make_stage(layer_num=4,in_channels=256,channels_expansion=2,stride_expansion=2,first_construct=False)
        self.stage3=self._make_stage(layer_num=6,in_channels=512,channels_expansion=2,stride_expansion=2,first_construct=False)
        self.stage4=self._make_stage(layer_num=3,in_channels=1024,channels_expansion=2,stride_expansion=2,first_construct=False)

        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc=nn.Linear(in_features=2048,out_features=num_classes)
        self.softmax=nn.Softmax(dim=1)

        self._init_weights()

    def _make_stage(self,layer_num,in_channels,channels_expansion,\
        stride_expansion,first_construct):

        layers=[]
        layers.append(self.conv_bottleneck(in_channels,channels_expansion,\
            stride_expansion,first_construct))
        for _ in range(layer_num-1):
            layers.append(self.identity_bottleneck(in_channels*channels_expansion))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1.0)
                nn.init.constant_(m.bias,0.0)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.stage4(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)
        x=self.softmax(x)

        return x
        
def Resnet50(num_classes=1000):
    model=ResNet(ConvBottleneck,IdentityBottleneck,num_classes)
    return model

#resnet50=Resnet50(num_classes=1000)