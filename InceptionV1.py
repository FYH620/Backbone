import torch
from torch import nn

class LRN(nn.Module):
    def __init__(self,k=2,alpha=0.0001,beta=0.75,local_size=5):
        super(LRN,self).__init__()
        self.k=k
        self.alpha=alpha
        self.beta=beta
        self.local_size=local_size
        self.average=nn.AvgPool3d(kernel_size=(local_size,1,1),stride=1,\
            padding=(int((local_size-1)/2),0,0))
    
    def forward(self,x):
        div=x.pow(2).unsqueeze(1)
        div=self.average(div).squeeze(1)
        div=(div.mul(self.alpha).add(self.k)).pow(self.beta)
        x=x.div(div)
        return x

class InceptionBlock(nn.Module):
    def __init__(self,in_dims,out_1_1_dims,out_2_1_dims,out_2_dims,out_3_1_dims,out_3_2_dims,out_4_2_dims):
        super(InceptionBlock,self).__init__()
        self.branch1x1=nn.Sequential(
            nn.Conv2d(in_dims,out_1_1_dims,kernel_size=1,stride=1),
            nn.ReLU()
        )
        self.branch3x3=nn.Sequential(
            nn.Conv2d(in_dims,out_2_1_dims,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(out_2_1_dims,out_2_dims,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.branch5x5=nn.Sequential(
            nn.Conv2d(in_dims,out_3_1_dims,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(out_3_1_dims,out_3_2_dims,kernel_size=5,stride=1,padding=2),
            nn.ReLU()
        )
        self.branch_pool=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_dims,out_4_2_dims,kernel_size=1,stride=1),
            nn.ReLU()
        )

    def forward(self,x):
        x1=self.branch1x1(x)
        x2=self.branch3x3(x)
        x3=self.branch5x5(x)
        x4=self.branch_pool(x)
        output=torch.cat([x1,x2,x3,x4],dim=1)
        return output

class InceptionV1(nn.Module):
    def __init__(self,num_classes):
        super(InceptionV1, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            LRN(k=2,alpha=0.0001,beta=0.75,local_size=11)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            LRN(k=2,alpha=0.0001,beta=0.75,local_size=11),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.block3 = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.block4 = nn.Sequential(
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.block5 = nn.Sequential(
            InceptionBlock(832, 256, 160, 320, 32, 128, 128),
            InceptionBlock(832, 384, 182, 384, 48, 128, 128),
        )
        self.gap=nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout=nn.Dropout(p=0.5)
        self.classifier=nn.Linear(1024,num_classes)

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.gap(x)
        x=x.view(x.shape[0], -1)
        x=self.dropout(x)
        x=self.classifier(x)
        return x
