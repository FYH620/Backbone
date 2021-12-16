import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from collections import OrderedDict

def conv3x3(in_dims,out_dims,stride=1,padding=1,groups=1,bias=False):
    return nn.Conv2d(in_dims,out_dims,kernel_size=3,stride=stride,\
        padding=padding,groups=groups,bias=bias)

def conv1x1(in_dims,out_dims,groups=1):
    return nn.Conv2d(in_dims,out_dims,kernel_size=1,stride=1,padding=0,groups=groups,bias=False)

def channel_shuffle(x,groups):
    batch_size,channels,height,width=x.shape
    channels_per_group=channels//groups
    x=x.view(batch_size,groups,channels_per_group,height,width)
    x=torch.transpose(x,1,2).contiguous()
    x=x.view(batch_size,-1,height,width)
    return x

class ShuffleUnit(nn.Module):
    def __init__(self,in_planes,out_planes,combine,groups,groups_conv=True):
        super(ShuffleUnit,self).__init__()
        self.groups=groups
        self.combine=combine
        planes=in_planes//4
        first_1x1conv_group=self.groups if groups_conv else 1
        if combine=='add':
            stride3x3=1
        elif combine=='concat':
            stride3x3=2
            self.avgpool=nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
            out_planes-=in_planes
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(combine))

        self.group_1x1_conv1=conv1x1(in_planes,planes,groups=first_1x1conv_group)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.depth_wise_conv3x3=conv3x3(planes,planes,groups=planes,stride=stride3x3)
        self.bn2=nn.BatchNorm2d(planes)
        self.group_1x1_conv2=conv1x1(planes,out_planes,groups=groups)
        self.bn3=nn.BatchNorm2d(out_planes)

    def forward(self,x):
        out=self.group_1x1_conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=channel_shuffle(out,groups=self.groups)
        out=self.depth_wise_conv3x3(out)
        out=self.bn2(out)
        out=self.group_1x1_conv2(out)
        out=self.bn3(out)
        
        if self.combine=='add':
            return F.relu(out+x)
        elif self.combine=='concat':
            x=self.avgpool(x)
            output=torch.cat([x,out],dim=1)
            return F.relu(output)

class ShuffleNet(nn.Module):
    def __init__(self,groups,num_classes=1000):
        super(ShuffleNet,self).__init__()
        self.stage_repeats = [3, 7, 3]
        self.groups=groups
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 576]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))
        
        self.conv1=conv3x3(3,self.stage_out_channels[1],stride=2)
        self.bn1=nn.BatchNorm2d(self.stage_out_channels[1])
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.stage2=self._make_stages(stage_index=2)
        self.stage3=self._make_stages(stage_index=3)
        self.stage4=self._make_stages(stage_index=4)

        self.fc=nn.Linear(self.stage_out_channels[-1],num_classes)
        self.gap=nn.AdaptiveAvgPool2d(output_size=1)
        self.init_weights()
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight,1)
                init.constant(m.bias,0)
            elif isinstance(m,nn.Linear):
                init.normal(m.weight,std=0.001)
                if m.bias is not None:
                    init.constant(m.bias,0)

    def _make_stages(self,stage_index):
        modules=OrderedDict()
        stage_name="ShuffleNet_Stage{}".format(stage_index)

        first_module=ShuffleUnit(
            in_planes=self.stage_out_channels[stage_index-1],
            out_planes=self.stage_out_channels[stage_index],
            groups=self.groups,
            groups_conv=stage_index>2,
            combine='concat'
        )
        modules[stage_name+"_0"]=first_module
        
        for i in range(self.stage_repeats[stage_index-2]):
            name=stage_name+"_"+str(i+1)
            one_module=ShuffleUnit(
                in_planes=self.stage_out_channels[stage_index],
                out_planes=self.stage_out_channels[stage_index],
                groups=self.groups,
                groups_conv=True,
                combine='add'
            )
            modules[name]=one_module
        
        return nn.Sequential(modules)

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.maxpool1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.stage4(x)
        x=self.gap(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return F.softmax(x,dim=1)

# shufflenet=ShuffleNet(groups=3,num_classes=1000)